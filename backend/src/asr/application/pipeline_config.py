from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from asr.application.ports import (
    AudioSegmenterFactoryPort,
    AsrBackendFactoryPort,
    AsrLoggerFactoryPort,
    RealtimeWorkerRunnerPort,
)
from asr.application.segmentation import SegmenterConfig
from asr.domain.types import Mode, OverloadStrategy
from diarization.application.diarization import DiarizationConfig, DiarizationRuntimeFactoryPort
from diarization.domain.types import DiarBackend


@dataclass(frozen=True)
class ASRPipelineDependencies:
    logger_factory: AsrLoggerFactoryPort
    asr_backend_factory: AsrBackendFactoryPort
    worker_runner: RealtimeWorkerRunnerPort
    diarization_factory: DiarizationRuntimeFactoryPort
    segmenter_factory: AudioSegmenterFactoryPort


@dataclass(frozen=True)
class ASRPipelineSettings:
    language: str = "ru"
    mode: Mode = "mix"
    source_names: Optional[List[str]] = None
    source_speaker_labels: Dict[str, str] = field(default_factory=dict)
    asr_model_name: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "int8_float16"
    cpu_threads: int = 0
    num_workers: int = 1
    beam_size: int = 5
    endpoint_silence_ms: float = 800.0
    max_segment_s: float = 12.0
    overlap_ms: float = 300.0
    vad_energy_threshold: float = 0.006
    vad_hangover_ms: int = 400
    vad_min_speech_ms: int = 350
    vad_band_ratio_min: float = 0.35
    vad_voiced_min: float = 0.12
    vad_pre_speech_ms: int = 120
    vad_min_end_silence_ms: int = 220
    min_segment_ms: int = 650
    diarization_enabled: bool = False
    diar_backend: DiarBackend = "pyannote"
    diar_sim_threshold: float = 0.74
    diar_min_segment_s: float = 1.0
    diar_window_s: float = 120.0
    diar_chunk_s: float = 30.0
    diar_step_s: float = 10.0
    diarization_sidecar_enabled: bool = True
    diarization_queue_size: int = 50
    diar_sherpa_embedding_model_path: str = ""
    diar_sherpa_provider: str = "cpu"
    diar_sherpa_num_threads: int = 1
    agc_enabled: bool = True
    agc_target_rms: float = 0.06
    agc_max_gain: float = 6.0
    agc_alpha: float = 0.02
    text_dedup_enabled: bool = True
    text_dedup_window: int = 80
    adaptive_beam_enabled: bool = True
    adaptive_beam_min: int = 1
    adaptive_beam_max: Optional[int] = None
    overload_enter_qsize: int = 18
    overload_exit_qsize: int = 6
    overload_hard_drop_qsize: int = 28
    overload_hold_s: float = 2.5
    overload_beam_cap: int = 2
    overload_overlap_ms: float = 120.0
    overload_max_segment_s: float = 5.0
    overload_strategy: OverloadStrategy = "drop_old"
    utterance_enabled: bool = True
    utterance_gap_s: float = 0.85
    utterance_max_s: float = 18.0
    utterance_flush_s: float = 2.5
    log_max_bytes: int = 25 * 1024 * 1024
    log_backup_count: int = 5
    log_speaker_labels: bool = True
    asr_language: Optional[str] = "ru"
    asr_initial_prompt: Optional[str] = None
    metrics_emit_interval_s: float = 1.0
    metrics_latency_window: int = 200

    @property
    def normalized_overload_strategy(self) -> OverloadStrategy:
        return "keep_all" if str(self.overload_strategy).strip().lower() == "keep_all" else "drop_old"

    @property
    def resolved_adaptive_beam_max(self) -> int:
        max_beam = int(self.adaptive_beam_max) if self.adaptive_beam_max is not None else int(self.beam_size)
        return max(1, max_beam)


def build_segmenter_config(settings: ASRPipelineSettings) -> SegmenterConfig:
    return SegmenterConfig(
        vad_energy_threshold=float(settings.vad_energy_threshold),
        vad_hangover_ms=int(settings.vad_hangover_ms),
        vad_min_speech_ms=int(settings.vad_min_speech_ms),
        vad_band_ratio_min=float(settings.vad_band_ratio_min),
        vad_voiced_min=float(settings.vad_voiced_min),
        vad_pre_speech_ms=int(settings.vad_pre_speech_ms),
        vad_min_end_silence_ms=int(settings.vad_min_end_silence_ms),
        min_segment_ms=int(settings.min_segment_ms),
        agc_enabled=bool(settings.agc_enabled),
        agc_target_rms=float(settings.agc_target_rms),
        agc_max_gain=float(settings.agc_max_gain),
        agc_alpha=float(settings.agc_alpha),
    )


def build_diarization_config(settings: ASRPipelineSettings, *, project_root: Optional[Any] = None) -> DiarizationConfig:
    temp_dir = None
    if project_root is not None:
        temp_dir = Path(project_root) / "tmp" / "nemo"
    return DiarizationConfig(
        enabled=bool(settings.diarization_enabled),
        backend=settings.diar_backend,
        sim_threshold=float(settings.diar_sim_threshold),
        min_segment_s=float(settings.diar_min_segment_s),
        window_s=float(settings.diar_window_s),
        chunk_s=float(settings.diar_chunk_s),
        step_s=float(settings.diar_step_s),
        device=str(settings.device),
        temp_dir=temp_dir,
        source_speaker_labels=dict(settings.source_speaker_labels or {}),
        sherpa_embedding_model_path=str(settings.diar_sherpa_embedding_model_path or ""),
        sherpa_provider=str(settings.diar_sherpa_provider or "cpu"),
        sherpa_num_threads=max(1, int(settings.diar_sherpa_num_threads)),
    )
