from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from audio.types import AudioSource


@dataclass
class SourceState:
    src: AudioSource
    enabled: bool = True

    buf: Deque[np.ndarray] = None  # type: ignore[assignment]
    buf_lock: threading.Lock = None  # type: ignore[assignment]

    delay_frames: int = 0
    delay_buf: Deque[np.ndarray] = None  # type: ignore[assignment]

    rms: float = 0.0
    last_ts: float = 0.0

    dropped_in_frames: int = 0
    missing_out_frames: int = 0
    buffer_frames: int = 0
    src_rate: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "buf", deque())
        object.__setattr__(self, "buf_lock", threading.Lock())
        object.__setattr__(self, "delay_buf", deque())
