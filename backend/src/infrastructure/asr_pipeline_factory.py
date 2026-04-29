from __future__ import annotations

from pathlib import Path
from typing import Any

from application.asr_session import ASRRuntime, ASRRuntimeFactory, ASRSessionSettings
from asr.application.pipeline import ASRPipeline
from asr.application.pipeline_config import ASRPipelineDependencies, ASRPipelineSettings
from asr.infrastructure.logger import ASRLogger
from asr.infrastructure.runtime_workers import ThreadRealtimeWorkerRunner
from asr.infrastructure.segmentation import AudioSegmenter
from asr.infrastructure.worker_faster_whisper import FasterWhisperASR
from diarization.infrastructure.diar_backend_pyannote import PyannoteDiarizer
from diarization.infrastructure.diarization_runtime import DefaultDiarizationRuntimeFactory
from diarization.infrastructure.diarizer import OnlineDiarizer


class ASRPipelineFactory(ASRRuntimeFactory):
    def build(
        self,
        settings: ASRSessionSettings,
        *,
        tap_queue: Any,
        project_root: Path,
        event_queue: Any = None,
    ) -> ASRRuntime:
        pipeline_settings = ASRPipelineSettings(
            language=settings.language,
            mode=settings.mode,
            source_speaker_labels=dict(settings.source_speaker_labels or {}),
            asr_model_name=settings.model_name,
            device=settings.device,
            compute_type=settings.compute_type,
            cpu_threads=settings.cpu_threads,
            num_workers=settings.num_workers,
            beam_size=settings.beam_size,
            endpoint_silence_ms=settings.endpoint_silence_ms,
            max_segment_s=settings.max_segment_s,
            overlap_ms=settings.overlap_ms,
            vad_energy_threshold=settings.vad_energy_threshold,
            vad_hangover_ms=350,
            vad_min_speech_ms=350,
            diarization_enabled=bool(settings.diarization_enabled),
            diar_backend=settings.diar_backend,
            diarization_sidecar_enabled=bool(settings.diarization_sidecar_enabled),
            diarization_queue_size=int(settings.diarization_queue_size),
            diar_sherpa_embedding_model_path=settings.diar_sherpa_embedding_model_path,
            diar_sherpa_provider=settings.diar_sherpa_provider,
            diar_sherpa_num_threads=settings.diar_sherpa_num_threads,
            log_speaker_labels=True,
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
            asr_language=settings.asr_language,
            asr_initial_prompt=settings.asr_initial_prompt,
            metrics_emit_interval_s=1.0,
            metrics_latency_window=200,
        )
        pipeline_dependencies = ASRPipelineDependencies(
            logger_factory=ASRLogger,
            asr_backend_factory=FasterWhisperASR,
            worker_runner=ThreadRealtimeWorkerRunner(),
            diarization_factory=DefaultDiarizationRuntimeFactory(
                online_diarizer_factory=OnlineDiarizer,
                pyannote_diarizer_factory=PyannoteDiarizer,
            ),
            segmenter_factory=AudioSegmenter,
        )
        return ASRPipeline(
            tap_queue=tap_queue,
            project_root=project_root,
            settings=pipeline_settings,
            dependencies=pipeline_dependencies,
            event_queue=event_queue,
        )
