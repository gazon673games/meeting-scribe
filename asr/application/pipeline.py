from __future__ import annotations

import queue
import threading
import time
from typing import List, Optional, Tuple

from asr.application.diarization import DiarizationRuntime
from asr.application.events import ASREventPublisher
from asr.application.ingest import TapIngestRuntime
from asr.application.metrics import ASRMetrics
from asr.application.overload import OverloadController
from asr.application.ports import (
    AsrBackendFactoryPort,
    AsrLoggerFactoryPort,
    OnlineDiarizerFactoryPort,
    PyannoteDiarizerFactoryPort,
)
from asr.application.policies import AdaptiveBeam
from asr.application.segmentation import AudioSegmenter, SegmenterConfig
from asr.application.transcription_worker import TranscriptionWorkerRuntime
from asr.application.utterances import UtteranceAggregator
from asr.domain import DiarBackend, Mode, OverloadStrategy, Segment


class ASRPipeline:
    """
    Thin lifecycle facade for realtime ASR.

    Domain/runtime responsibilities live in focused collaborators:
    segmentation, ingest, overload, metrics, event publishing, diarization,
    utterance aggregation, and transcription worker runtime.
    """

    def __init__(
        self,
        *,
        tap_queue: "queue.Queue[dict]",
        project_root,
        logger_factory: AsrLoggerFactoryPort,
        asr_backend_factory: AsrBackendFactoryPort,
        online_diarizer_factory: OnlineDiarizerFactoryPort,
        pyannote_diarizer_factory: PyannoteDiarizerFactoryPort,
        language: str = "ru",
        mode: Mode = "mix",
        source_names: Optional[List[str]] = None,
        asr_model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        beam_size: int = 5,
        ui_queue: Optional["queue.Queue[dict]"] = None,
        event_queue: Optional["queue.Queue[dict]"] = None,
        endpoint_silence_ms: float = 800.0,
        max_segment_s: float = 12.0,
        overlap_ms: float = 300.0,
        vad_energy_threshold: float = 0.006,
        vad_hangover_ms: int = 400,
        vad_min_speech_ms: int = 350,
        vad_band_ratio_min: float = 0.35,
        vad_voiced_min: float = 0.12,
        vad_pre_speech_ms: int = 120,
        vad_min_end_silence_ms: int = 220,
        min_segment_ms: int = 650,
        diarization_enabled: bool = True,
        diar_backend: DiarBackend = "pyannote",
        diar_sim_threshold: float = 0.74,
        diar_min_segment_s: float = 1.0,
        diar_window_s: float = 120.0,
        diar_chunk_s: float = 30.0,
        diar_step_s: float = 10.0,
        agc_enabled: bool = True,
        agc_target_rms: float = 0.06,
        agc_max_gain: float = 6.0,
        agc_alpha: float = 0.02,
        text_dedup_enabled: bool = True,
        text_dedup_window: int = 80,
        adaptive_beam_enabled: bool = True,
        adaptive_beam_min: int = 1,
        adaptive_beam_max: Optional[int] = None,
        overload_enter_qsize: int = 18,
        overload_exit_qsize: int = 6,
        overload_hard_drop_qsize: int = 28,
        overload_hold_s: float = 2.5,
        overload_beam_cap: int = 2,
        overload_overlap_ms: float = 120.0,
        overload_max_segment_s: float = 5.0,
        overload_strategy: OverloadStrategy = "drop_old",
        utterance_enabled: bool = True,
        utterance_gap_s: float = 0.85,
        utterance_max_s: float = 18.0,
        utterance_flush_s: float = 2.5,
        log_max_bytes: int = 25 * 1024 * 1024,
        log_backup_count: int = 5,
        log_speaker_labels: bool = True,
        asr_language: Optional[str] = "ru",
        asr_initial_prompt: Optional[str] = None,
        metrics_emit_interval_s: float = 1.0,
        metrics_latency_window: int = 200,
    ) -> None:
        self.tap_q = tap_queue
        self.project_root = project_root
        self.language = language
        self.mode = mode
        self.source_names = source_names

        self.asr_model_name = asr_model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = int(beam_size)
        self.asr_language = asr_language
        self.asr_initial_prompt = asr_initial_prompt

        self._endpoint_silence_ms_base = float(endpoint_silence_ms)
        self._max_segment_s_base = float(max_segment_s)
        self._overlap_ms_base = float(overlap_ms)

        self._stop = threading.Event()
        self._ingest_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._seg_q: "queue.Queue[Segment]" = queue.Queue(maxsize=50)

        self.session_id = f"sess_{int(time.time())}"
        self.logger = logger_factory(
            root=self.project_root,
            session_id=self.session_id,
            language=language,
            max_bytes=int(log_max_bytes),
            backup_count=int(log_backup_count),
        )
        self._events = ASREventPublisher(
            logger=self.logger,
            event_queue=event_queue if event_queue is not None else ui_queue,
        )

        self._over = OverloadController(
            enter_qsize=int(overload_enter_qsize),
            exit_qsize=int(overload_exit_qsize),
            hard_qsize=int(overload_hard_drop_qsize),
            hold_s=float(overload_hold_s),
            beam_cap=max(1, int(overload_beam_cap)),
            overlap_ms=float(overload_overlap_ms),
            max_segment_s=float(overload_max_segment_s),
            strategy="keep_all" if str(overload_strategy).strip().lower() == "keep_all" else "drop_old",
        )

        self._diar = DiarizationRuntime(
            enabled=bool(diarization_enabled),
            backend=diar_backend,
            sim_threshold=float(diar_sim_threshold),
            min_segment_s=float(diar_min_segment_s),
            window_s=float(diar_window_s),
            chunk_s=float(diar_chunk_s),
            step_s=float(diar_step_s),
            device=self.device,
            online_diarizer_factory=online_diarizer_factory,
            pyannote_diarizer_factory=pyannote_diarizer_factory,
        )

        max_beam = int(adaptive_beam_max) if adaptive_beam_max is not None else int(self.beam_size)
        max_beam = max(1, max_beam)
        self._beam_ctl = AdaptiveBeam(
            min_beam=max(1, int(adaptive_beam_min)),
            max_beam=max_beam,
            cur_beam=max(1, min(int(self.beam_size), max_beam)),
        )

        self._utt = UtteranceAggregator(
            enabled=bool(utterance_enabled),
            gap_s=float(utterance_gap_s),
            max_s=float(utterance_max_s),
            flush_s=float(utterance_flush_s),
            log_speaker_labels=bool(log_speaker_labels),
        )

        self._metrics = ASRMetrics(
            latency_window=int(metrics_latency_window),
            emit_interval_s=float(metrics_emit_interval_s),
        )

        self._segmenter_config = SegmenterConfig(
            vad_energy_threshold=float(vad_energy_threshold),
            vad_hangover_ms=int(vad_hangover_ms),
            vad_min_speech_ms=int(vad_min_speech_ms),
            vad_band_ratio_min=float(vad_band_ratio_min),
            vad_voiced_min=float(vad_voiced_min),
            vad_pre_speech_ms=int(vad_pre_speech_ms),
            vad_min_end_silence_ms=int(vad_min_end_silence_ms),
            min_segment_ms=int(min_segment_ms),
            agc_enabled=bool(agc_enabled),
            agc_target_rms=float(agc_target_rms),
            agc_max_gain=float(agc_max_gain),
            agc_alpha=float(agc_alpha),
        )

        self._segmenter = AudioSegmenter(
            config=self._segmenter_config,
            segment_queue=self._seg_q,
            diarization=self._diar,
            metrics=self._metrics,
            log_event=self._events.log,
            segmentation_params=self._segmentation_params,
        )

        self._worker = TranscriptionWorkerRuntime(
            segment_queue=self._seg_q,
            stop_event=self._stop,
            log_event=self._events.log,
            metrics=self._metrics,
            overload=self._over,
            beam_controller=self._beam_ctl,
            diarization=self._diar,
            utterances=self._utt,
            model_name=self.asr_model_name,
            language=self.asr_language,
            device=self.device,
            compute_type=self.compute_type,
            beam_size=self.beam_size,
            initial_prompt=self.asr_initial_prompt,
            asr_backend_factory=asr_backend_factory,
            text_dedup_enabled=bool(text_dedup_enabled),
            text_dedup_window=int(text_dedup_window),
            adaptive_beam_enabled=bool(adaptive_beam_enabled),
            log_speaker_labels=bool(log_speaker_labels),
        )

        self._ingest = TapIngestRuntime(
            tap_queue=self.tap_q,
            stop_event=self._stop,
            mode=self.mode,
            segmenter=self._segmenter,
            log_event=self._events.log,
            emit_metrics=lambda force=False: self._worker.emit_metrics(force=bool(force)),
        )

        self._text_dedup_enabled = bool(text_dedup_enabled)
        self._adaptive_beam_enabled = bool(adaptive_beam_enabled)
        self._log_speaker_labels = bool(log_speaker_labels)

    def start(self) -> None:
        self._stop.clear()
        self._over.reset()
        self._metrics.reset()
        self._segmenter.reset_runtime()
        self._worker.reset_runtime()

        self._events.log(self._start_event())

        self._ingest_thread = threading.Thread(target=self._ingest.run_safe, name="asr-ingest", daemon=True)
        self._worker_thread = threading.Thread(target=self._worker.run_safe, name="asr-worker", daemon=True)
        self._ingest_thread.start()
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ingest_thread:
            self._ingest_thread.join()
            self._ingest_thread = None
        if self._worker_thread:
            self._worker_thread.join()
            self._worker_thread = None

        try:
            self._worker.flush_utterances(force=True)
        except Exception:
            pass

        self._worker.emit_metrics(force=True)
        self._events.log({"type": "asr_stopped", "ts": time.time()})
        self.logger.close()

    def _segmentation_params(self) -> Tuple[float, float, float]:
        return self._over.segmentation_params(
            endpoint_silence_ms=self._endpoint_silence_ms_base,
            max_segment_s=self._max_segment_s_base,
            overlap_ms=self._overlap_ms_base,
        )

    def _start_event(self) -> dict:
        cfg = self._segmenter_config
        return {
            "type": "asr_started",
            "session_id": self.session_id,
            "language": self.language,
            "mode": self.mode,
            "model": self.asr_model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "endpoint_silence_ms": self._endpoint_silence_ms_base,
            "max_segment_s": self._max_segment_s_base,
            "overlap_ms": self._overlap_ms_base,
            "overload_strategy": self._over.strategy,
            "vad": {
                "energy_threshold": cfg.vad_energy_threshold,
                "hangover_ms": cfg.vad_hangover_ms,
                "min_speech_ms": cfg.vad_min_speech_ms,
                "band_ratio_min": cfg.vad_band_ratio_min,
                "voiced_min": cfg.vad_voiced_min,
                "pre_speech_ms": cfg.vad_pre_speech_ms,
                "min_end_silence_ms": cfg.vad_min_end_silence_ms,
                "min_segment_ms": cfg.min_segment_ms,
            },
            "overload": {
                "enter_qsize": self._over.enter_qsize,
                "exit_qsize": self._over.exit_qsize,
                "hard_drop_qsize": self._over.hard_qsize,
                "hold_s": self._over.hold_s,
                "beam_cap": self._over.beam_cap,
                "overlap_ms": self._over.overlap_ms,
                "max_segment_s": self._over.max_segment_s,
            },
            "diarization_enabled": self._diar.enabled,
            "diar_backend": self._diar.backend,
            "agc_enabled": cfg.agc_enabled,
            "text_dedup_enabled": self._text_dedup_enabled,
            "adaptive_beam_enabled": self._adaptive_beam_enabled,
            "utterance_enabled": self._utt.enabled,
            "utterance_gap_s": self._utt.gap_s,
            "utterance_max_s": self._utt.max_s,
            "utterance_flush_s": self._utt.flush_s,
            "log_rotation": {"max_bytes": self.logger.max_bytes, "backup_count": self.logger.backup_count},
            "log_speaker_labels": self._log_speaker_labels,
            "asr_language": self.asr_language,
            "asr_initial_prompt": bool(self.asr_initial_prompt),
            "ts": time.time(),
        }
