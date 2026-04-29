from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from asr.domain.types import Mode, OverloadStrategy
from diarization.domain.types import DiarBackend


class ASRRuntime(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


@dataclass(frozen=True)
class ASRSessionSettings:
    language: str
    mode: Mode
    model_name: str
    device: str
    compute_type: str
    cpu_threads: int
    num_workers: int
    beam_size: int
    endpoint_silence_ms: float
    max_segment_s: float
    overlap_ms: float
    vad_energy_threshold: float
    overload_strategy: OverloadStrategy
    overload_enter_qsize: int
    overload_exit_qsize: int
    overload_hard_qsize: int
    overload_beam_cap: int
    overload_max_segment_s: float
    overload_overlap_ms: float
    asr_language: Optional[str]
    asr_initial_prompt: Optional[str]
    source_speaker_labels: Dict[str, str] = field(default_factory=dict)
    diarization_enabled: bool = False
    diar_backend: DiarBackend = "online"
    diarization_sidecar_enabled: bool = True
    diarization_queue_size: int = 50
    diar_sherpa_embedding_model_path: str = ""
    diar_sherpa_provider: str = "cpu"
    diar_sherpa_num_threads: int = 1


class ASRRuntimeFactory(Protocol):
    def build(
        self,
        settings: ASRSessionSettings,
        *,
        tap_queue: Any,
        project_root: Path,
        event_queue: Any = None,
    ) -> ASRRuntime:
        ...
