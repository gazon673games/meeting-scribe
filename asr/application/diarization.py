from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from asr.domain.segments import Segment
from asr.domain.types import DiarBackend

LogEvent = Callable[[dict], None]


@dataclass(frozen=True)
class DiarizationConfig:
    enabled: bool
    backend: DiarBackend
    sim_threshold: float
    min_segment_s: float
    window_s: float
    chunk_s: float
    step_s: float
    device: str


class DiarizationPort(Protocol):
    enabled: bool
    backend: DiarBackend

    def ensure_stream(self, name: str) -> None:
        ...

    def update_ring(self, stream: str, t1: float, audio_16k: Any) -> None:
        ...

    def init_backend(self, log_event: LogEvent) -> None:
        ...

    def speaker_for_segment(self, seg: Segment, log_event: LogEvent) -> str:
        ...


class DiarizationRuntimeFactoryPort(Protocol):
    def __call__(self, *, config: DiarizationConfig) -> DiarizationPort:
        ...
