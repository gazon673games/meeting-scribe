from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Protocol

Mode = Literal["mix", "split"]
DiarBackend = Literal["pyannote", "online", "nemo"]
OverloadStrategy = Literal["drop_old", "keep_all"]


class MonoAudio16k(Protocol):
    @property
    def sample_rate_hz(self) -> int:
        ...

    @property
    def frame_count(self) -> int:
        ...

    @property
    def duration_s(self) -> float:
        ...

    @property
    def samples(self) -> object:
        ...


@dataclass
class Segment:
    stream: str
    t_start: float
    t_end: float
    audio: MonoAudio16k
    enqueue_ts: float


@dataclass
class DiarSegment:
    t0: float
    t1: float
    speaker: str


@dataclass
class UtteranceState:
    stream: str
    speaker: str
    t_start: float
    t_end: float
    text: str
    last_emit_ts: float


def pick_speaker(timeline: List[DiarSegment], t0: float, t1: float) -> str:
    best_label = "S?"
    best_ov = 0.0
    for segment in timeline:
        a = max(float(t0), float(segment.t0))
        b = min(float(t1), float(segment.t1))
        ov = b - a
        if ov > best_ov:
            best_ov = ov
            best_label = str(segment.speaker)
    return best_label if best_ov > 0 else "S?"
