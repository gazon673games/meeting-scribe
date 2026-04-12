from __future__ import annotations

from pathlib import Path
from typing import Any

from application.asr_session import ASRRuntime, ASRRuntimeFactory, ASRSessionSettings
from asr.application.pipeline import ASRPipeline
from asr.infrastructure.diar_backend_pyannote import PyannoteDiarizer
from asr.infrastructure.diarizer import OnlineDiarizer
from asr.infrastructure.logger import ASRLogger
from asr.infrastructure.worker_faster_whisper import FasterWhisperASR


class ASRPipelineFactory(ASRRuntimeFactory):
    def build(
        self,
        settings: ASRSessionSettings,
        *,
        tap_queue: Any,
        project_root: Path,
        event_queue: Any = None,
    ) -> ASRRuntime:
        return ASRPipeline(
            tap_queue=tap_queue,
            project_root=project_root,
            logger_factory=ASRLogger,
            asr_backend_factory=FasterWhisperASR,
            online_diarizer_factory=OnlineDiarizer,
            pyannote_diarizer_factory=PyannoteDiarizer,
            language=settings.language,
            mode=settings.mode,
            asr_model_name=settings.model_name,
            device=settings.device,
            compute_type=settings.compute_type,
            beam_size=settings.beam_size,
            endpoint_silence_ms=settings.endpoint_silence_ms,
            max_segment_s=settings.max_segment_s,
            overlap_ms=settings.overlap_ms,
            vad_energy_threshold=settings.vad_energy_threshold,
            vad_hangover_ms=350,
            vad_min_speech_ms=350,
            diarization_enabled=False,
            log_speaker_labels=False,
            overload_strategy=settings.overload_strategy,
            overload_enter_qsize=settings.overload_enter_qsize,
            overload_exit_qsize=settings.overload_exit_qsize,
            overload_hard_drop_qsize=settings.overload_hard_qsize,
            overload_hold_s=2.5,
            overload_beam_cap=settings.overload_beam_cap,
            overload_overlap_ms=settings.overload_overlap_ms,
            overload_max_segment_s=settings.overload_max_segment_s,
            utterance_enabled=True,
            utterance_gap_s=0.85,
            utterance_max_s=18.0,
            utterance_flush_s=2.5,
            log_max_bytes=25 * 1024 * 1024,
            log_backup_count=5,
            event_queue=event_queue,
            asr_language=settings.asr_language,
            asr_initial_prompt=settings.asr_initial_prompt,
            metrics_emit_interval_s=1.0,
            metrics_latency_window=200,
        )
