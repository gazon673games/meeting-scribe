from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VADDecision:
    speech: bool
    energy_rms: float
    thr: float
    noise_rms: float
    band_ratio: float
    voiced: float
