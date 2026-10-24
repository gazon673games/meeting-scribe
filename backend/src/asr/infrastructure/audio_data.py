from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class MonoAudio16kBuffer:
    _samples: np.ndarray
    sample_rate_hz: int = 16000

    @classmethod
    def from_array(
        cls, samples, *, sample_rate_hz: int = 16000
    ) -> MonoAudio16kBuffer:
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        return cls(_samples=data, sample_rate_hz=int(sample_rate_hz))

    @property
    def samples(self) -> np.ndarray:
        return self._samples

    @property
    def frame_count(self) -> int:
        return int(self._samples.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.frame_count) / float(max(1, self.sample_rate_hz))


@dataclass(frozen=True)
class ChunkedAudio16kBuffer:
    """Retain frame views and materialize contiguous samples only on demand."""

    chunks: tuple[np.ndarray, ...]
    sample_rate_hz: int = 16000

    @cached_property
    def samples(self) -> np.ndarray:
        if not self.chunks:
            return np.empty(0, dtype=np.float32)
        if len(self.chunks) == 1:
            return self.chunks[0].copy()
        return np.concatenate(self.chunks).astype(np.float32, copy=False)

    @cached_property
    def frame_count(self) -> int:
        return sum(int(chunk.size) for chunk in self.chunks)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.sample_rate_hz

    def tail(self, start: int) -> ChunkedAudio16kBuffer:
        remaining = max(0, int(start))
        for index, chunk in enumerate(self.chunks):
            if remaining < chunk.size:
                return ChunkedAudio16kBuffer(
                    (chunk[remaining:], *self.chunks[index + 1 :]),
                    self.sample_rate_hz,
                )
            remaining -= int(chunk.size)
        return ChunkedAudio16kBuffer((), self.sample_rate_hz)
