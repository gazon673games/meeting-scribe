from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from asr.application.diarization import DiarizationPort
from asr.application.events import ASREventPublisher
from asr.application.ingest import TapIngestRuntime
from asr.application.metrics import ASRMetrics
from asr.application.overload import OverloadController
from asr.application.pipeline_config import (
    ASRPipelineDependencies,
    ASRPipelineSettings,
    build_diarization_config,
    build_segmenter_config,
)
from asr.application.policies import AdaptiveBeam
from asr.application.ports import AsrLoggerPort, RealtimeWorkerRunnerPort, StopSignalPort, WorkerHandlePort
from asr.application.segmentation import AudioSegmenterPort, SegmenterConfig
from asr.application.transcription_worker import TranscriptionWorkerRuntime
from asr.application.utterances import UtteranceAggregator
from asr.domain import Segment


@dataclass
class ASRRuntimeGraph:
    worker_runner: RealtimeWorkerRunnerPort
    stop: StopSignalPort
    logger: AsrLoggerPort
    events: ASREventPublisher
    overload: OverloadController
    diarization: DiarizationPort
    utterances: UtteranceAggregator
    metrics: ASRMetrics
    segmenter_config: SegmenterConfig
    segmenter: AudioSegmenterPort
    worker: TranscriptionWorkerRuntime
    ingest: TapIngestRuntime
    ingest_worker: Optional[WorkerHandlePort] = None
    transcription_worker: Optional[WorkerHandlePort] = None

    def start(self, *, settings: ASRPipelineSettings, session_id: str) -> None:
        self.stop.clear()
        self.overload.reset()
        self.metrics.reset()
        self.segmenter.reset_runtime()
        self.worker.reset_runtime()

        self.events.log(self.start_event(settings=settings, session_id=session_id))
        self.ingest_worker = self.worker_runner.start_worker(name="asr-ingest", target=self.ingest.run_safe)
        self.transcription_worker = self.worker_runner.start_worker(
            name="asr-worker",
            target=self.worker.run_safe,
        )

    def stop_runtime(self) -> None:
        self.stop.set()
        if self.ingest_worker:
            self.ingest_worker.join()
            self.ingest_worker = None
        if self.transcription_worker:
            self.transcription_worker.join()
            self.transcription_worker = None

        try:
            self.worker.flush_utterances(force=True)
        except Exception:
            pass

        self.worker.emit_metrics(force=True)
        self.events.log({"type": "asr_stopped", "ts": time.time()})
        self.logger.close()

    def segmentation_params(self, settings: ASRPipelineSettings) -> Tuple[float, float, float]:
        return self.overload.segmentation_params(
            endpoint_silence_ms=settings.endpoint_silence_ms,
            max_segment_s=settings.max_segment_s,
            overlap_ms=settings.overlap_ms,
        )

    def start_event(self, *, settings: ASRPipelineSettings, session_id: str) -> dict:
        cfg = self.segmenter_config
        return {
            "type": "asr_started",
            "session_id": session_id,
            "language": settings.language,
            "mode": settings.mode,
            "model": settings.asr_model_name,
            "device": settings.device,
            "compute_type": settings.compute_type,
            "beam_size": int(settings.beam_size),
            "endpoint_silence_ms": settings.endpoint_silence_ms,
            "max_segment_s": settings.max_segment_s,
            "overlap_ms": settings.overlap_ms,
            "overload_strategy": self.overload.strategy,
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
                "enter_qsize": self.overload.enter_qsize,
                "exit_qsize": self.overload.exit_qsize,
                "hard_drop_qsize": self.overload.hard_qsize,
                "hold_s": self.overload.hold_s,
                "beam_cap": self.overload.beam_cap,
                "overlap_ms": self.overload.overlap_ms,
                "max_segment_s": self.overload.max_segment_s,
            },
            "diarization_enabled": self.diarization.enabled,
            "diar_backend": self.diarization.backend,
            "agc_enabled": cfg.agc_enabled,
            "text_dedup_enabled": settings.text_dedup_enabled,
            "adaptive_beam_enabled": settings.adaptive_beam_enabled,
            "utterance_enabled": self.utterances.enabled,
            "utterance_gap_s": self.utterances.gap_s,
            "utterance_max_s": self.utterances.max_s,
            "utterance_flush_s": self.utterances.flush_s,
            "log_rotation": {"max_bytes": self.logger.max_bytes, "backup_count": self.logger.backup_count},
            "log_speaker_labels": settings.log_speaker_labels,
            "asr_language": settings.asr_language,
            "asr_initial_prompt": bool(settings.asr_initial_prompt),
            "ts": time.time(),
        }


def build_runtime_graph(
    *,
    settings: ASRPipelineSettings,
    dependencies: ASRPipelineDependencies,
    tap_queue: "queue.Queue[dict]",
    project_root: Any,
    session_id: str,
    ui_queue: Optional["queue.Queue[dict]"] = None,
    event_queue: Optional["queue.Queue[dict]"] = None,
) -> ASRRuntimeGraph:
    stop = dependencies.worker_runner.create_stop_signal()
    segment_queue: "queue.Queue[Segment]" = queue.Queue(maxsize=50)
    logger = dependencies.logger_factory(
        root=project_root,
        session_id=session_id,
        language=settings.language,
        max_bytes=int(settings.log_max_bytes),
        backup_count=int(settings.log_backup_count),
    )
    events = ASREventPublisher(logger=logger, event_queue=event_queue if event_queue is not None else ui_queue)
    overload = OverloadController(
        enter_qsize=int(settings.overload_enter_qsize),
        exit_qsize=int(settings.overload_exit_qsize),
        hard_qsize=int(settings.overload_hard_drop_qsize),
        hold_s=float(settings.overload_hold_s),
        beam_cap=max(1, int(settings.overload_beam_cap)),
        overlap_ms=float(settings.overload_overlap_ms),
        max_segment_s=float(settings.overload_max_segment_s),
        strategy=settings.normalized_overload_strategy,
    )
    diarization = dependencies.diarization_factory(config=build_diarization_config(settings))
    max_beam = settings.resolved_adaptive_beam_max
    beam_controller = AdaptiveBeam(
        min_beam=max(1, int(settings.adaptive_beam_min)),
        max_beam=max_beam,
        cur_beam=max(1, min(int(settings.beam_size), max_beam)),
    )
    utterances = UtteranceAggregator(
        enabled=bool(settings.utterance_enabled),
        gap_s=float(settings.utterance_gap_s),
        max_s=float(settings.utterance_max_s),
        flush_s=float(settings.utterance_flush_s),
        log_speaker_labels=bool(settings.log_speaker_labels),
    )
    metrics = ASRMetrics(
        latency_window=int(settings.metrics_latency_window),
        emit_interval_s=float(settings.metrics_emit_interval_s),
    )
    segmenter_config = build_segmenter_config(settings)

    def segmentation_params() -> Tuple[float, float, float]:
        return overload.segmentation_params(
            endpoint_silence_ms=settings.endpoint_silence_ms,
            max_segment_s=settings.max_segment_s,
            overlap_ms=settings.overlap_ms,
        )

    segmenter = dependencies.segmenter_factory(
        config=segmenter_config,
        segment_queue=segment_queue,
        diarization=diarization,
        metrics=metrics,
        log_event=events.log,
        segmentation_params=segmentation_params,
    )
    worker = TranscriptionWorkerRuntime(
        segment_queue=segment_queue,
        stop_event=stop,
        log_event=events.log,
        metrics=metrics,
        overload=overload,
        beam_controller=beam_controller,
        diarization=diarization,
        utterances=utterances,
        model_name=settings.asr_model_name,
        language=settings.asr_language,
        device=settings.device,
        compute_type=settings.compute_type,
        beam_size=int(settings.beam_size),
        initial_prompt=settings.asr_initial_prompt,
        asr_backend_factory=dependencies.asr_backend_factory,
        text_dedup_enabled=bool(settings.text_dedup_enabled),
        text_dedup_window=int(settings.text_dedup_window),
        adaptive_beam_enabled=bool(settings.adaptive_beam_enabled),
        log_speaker_labels=bool(settings.log_speaker_labels),
    )
    ingest = TapIngestRuntime(
        tap_queue=tap_queue,
        stop_event=stop,
        mode=settings.mode,
        segmenter=segmenter,
        log_event=events.log,
        emit_metrics=lambda force=False: worker.emit_metrics(force=bool(force)),
    )
    return ASRRuntimeGraph(
        worker_runner=dependencies.worker_runner,
        stop=stop,
        logger=logger,
        events=events,
        overload=overload,
        diarization=diarization,
        utterances=utterances,
        metrics=metrics,
        segmenter_config=segmenter_config,
        segmenter=segmenter,
        worker=worker,
        ingest=ingest,
    )
