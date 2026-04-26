from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MonoAudio16kBuffer:
    _samples: np.ndarray
    sample_rate_hz: int = 16000

    @classmethod
    def from_array(cls, samples, *, sample_rate_hz: int = 16000) -> "MonoAudio16kBuffer":
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
