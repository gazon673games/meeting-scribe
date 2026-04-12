from __future__ import annotations

import numpy as np


class PreGainAGC:
    def __init__(self, target_rms: float = 0.06, max_gain: float = 6.0, alpha: float = 0.02):
        self.target_rms = float(target_rms)
        self.max_gain = float(max_gain)
        self.alpha = float(alpha)
        self.gain = 1.0
        self.last_in_rms = 0.0

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        xf = x.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(xf * xf)))

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        rms = self._rms(x)
        self.last_in_rms = rms
        if rms > 1e-7:
            desired = self.target_rms / rms
            desired = max(1.0 / self.max_gain, min(self.max_gain, desired))
            self.gain = (1.0 - self.alpha) * self.gain + self.alpha * desired
        y = x * float(self.gain)
        return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)
