from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from asr.application.pipeline_config import ASRPipelineSettings


@dataclass(frozen=True)
class StreamingWorkerConfig:
    model_name: str
    language: Optional[str]
    device: str
    compute_type: str
    cpu_threads: int
    num_workers: int
    initial_prompt: Optional[str]
    lookahead: int = 2
    queue_timeout_s: float = 0.1

    @classmethod
    def from_settings(cls, settings: "ASRPipelineSettings") -> "StreamingWorkerConfig":
        return cls(
            model_name=str(settings.asr_model_name),
            language=settings.asr_language,
            device=str(settings.device),
            compute_type=str(settings.compute_type),
            cpu_threads=int(settings.cpu_threads),
            num_workers=int(settings.num_workers),
            initial_prompt=settings.asr_initial_prompt,
        )
