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
    enqueue_monotonic: float | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.t_end) - float(self.t_start))

    def queue_wait_s(
        self, now: float, *, monotonic_now: float | None = None
    ) -> float:
        if self.enqueue_monotonic is not None and monotonic_now is not None:
            return max(0.0, monotonic_now - self.enqueue_monotonic)
        return max(0.0, float(now) - float(self.enqueue_ts))
