from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int
    channels: int
    dtype: str = "float32"
    blocksize: int = 1024
