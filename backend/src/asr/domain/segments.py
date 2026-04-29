from __future__ import annotations

from dataclasses import dataclass

from asr.domain.audio import MonoAudio16k


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
