from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

from asr.domain import Mode, OverloadStrategy


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
