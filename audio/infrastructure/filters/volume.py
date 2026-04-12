from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio.domain import AudioFormat


@dataclass
class VolumeFilter:
    gain: float = 1.0

    def process(self, x: np.ndarray, fmt: AudioFormat) -> np.ndarray:
        # preserve dtype float32
        return x.astype(np.float32, copy=False) * float(self.gain)
