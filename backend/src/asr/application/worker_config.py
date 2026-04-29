from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from asr.application.pipeline_config import ASRPipelineSettings


@dataclass(frozen=True)
class TranscriptionWorkerConfig:
    model_name: str
    language: Optional[str]
    device: str
    compute_type: str
    cpu_threads: int
    num_workers: int
    beam_size: int
    initial_prompt: Optional[str]
    text_dedup_enabled: bool
    text_dedup_window: int
    adaptive_beam_enabled: bool
    log_speaker_labels: bool
    init_diarization: bool
    diarization_blocking_lookup: bool
    source_speaker_labels: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_settings(cls, settings: ASRPipelineSettings) -> TranscriptionWorkerConfig:
        return cls(
            model_name=str(settings.asr_model_name),
            language=settings.asr_language,
            device=str(settings.device),
            compute_type=str(settings.compute_type),
            cpu_threads=int(settings.cpu_threads),
            num_workers=int(settings.num_workers),
            beam_size=int(settings.beam_size),
            initial_prompt=settings.asr_initial_prompt,
            text_dedup_enabled=bool(settings.text_dedup_enabled),
            text_dedup_window=int(settings.text_dedup_window),
            adaptive_beam_enabled=bool(settings.adaptive_beam_enabled),
            log_speaker_labels=bool(settings.log_speaker_labels),
            init_diarization=bool(settings.diarization_enabled and not settings.diarization_sidecar_enabled),
            diarization_blocking_lookup=bool(settings.diarization_enabled and not settings.diarization_sidecar_enabled),
            source_speaker_labels=dict(settings.source_speaker_labels or {}),
        )
