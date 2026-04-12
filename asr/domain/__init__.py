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

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.t_end) - float(self.t_start))

    def queue_wait_s(self, now: float) -> float:
        return max(0.0, float(now) - float(self.enqueue_ts))


@dataclass
class DiarSegment:
    t0: float
    t1: float
    speaker: str

    def overlap_duration(self, t0: float, t1: float) -> float:
        start = max(float(t0), float(self.t0))
        end = min(float(t1), float(self.t1))
        return max(0.0, end - start)


@dataclass
class UtteranceState:
    stream: str
    speaker: str
    t_start: float
    t_end: float
    text: str
    last_emit_ts: float

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.t_end) - float(self.t_start))

    def can_extend(self, *, t_start: float, t_end: float, gap_s: float, max_s: float) -> bool:
        gap = float(t_start) - float(self.t_end)
        new_duration = float(t_end) - float(self.t_start)
        return gap <= float(gap_s) and new_duration <= float(max_s)

    def extend(self, *, t_end: float, text: str, last_emit_ts: float) -> None:
        self.t_end = float(t_end)
        self.text = (str(self.text).strip() + " " + str(text).strip()).strip()
        self.last_emit_ts = float(last_emit_ts)

    def should_flush(self, *, now: float, flush_s: float, force: bool) -> bool:
        return bool(force) or (float(now) - float(self.last_emit_ts)) >= float(flush_s)


def pick_speaker(timeline: List[DiarSegment], t0: float, t1: float) -> str:
    best_label = "S?"
    best_ov = 0.0
    for segment in timeline:
        ov = segment.overlap_duration(t0, t1)
        if ov > best_ov:
            best_ov = ov
            best_label = str(segment.speaker)
    return best_label if best_ov > 0 else "S?"
