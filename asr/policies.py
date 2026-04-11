from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

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


@dataclass
class AdaptiveBeam:
    min_beam: int = 1
    max_beam: int = 5
    cur_beam: int = 5

    backlog_hi: int = 12
    backlog_lo: int = 2
    latency_ratio_hi: float = 1.1
    latency_ratio_lo: float = 0.7

    cool_down_s: float = 2.0
    last_change_ts: float = 0.0

    def maybe_update(
        self, *, seg_qsize: int, last_latency_s: float, last_dur_s: float, now: float
    ) -> Tuple[int, Optional[str]]:
        if (now - float(self.last_change_ts)) < float(self.cool_down_s):
            return (int(self.cur_beam), None)

        dur = max(1e-6, float(last_dur_s))
        ratio = float(last_latency_s) / dur

        reason = None
        if seg_qsize >= int(self.backlog_hi) or ratio >= float(self.latency_ratio_hi):
            if self.cur_beam > self.min_beam:
                self.cur_beam -= 1
                self.last_change_ts = now
                reason = f"downshift (q={seg_qsize}, lat_ratio={ratio:.2f})"
        elif seg_qsize <= int(self.backlog_lo) and ratio <= float(self.latency_ratio_lo):
            if self.cur_beam < self.max_beam:
                self.cur_beam += 1
                self.last_change_ts = now
                reason = f"upshift (q={seg_qsize}, lat_ratio={ratio:.2f})"

        return (int(self.cur_beam), reason)
