from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from audio.domain.ports import AudioSource


@dataclass
class SourceState:
    src: AudioSource
    enabled: bool = True

    buf: Deque[np.ndarray] = field(default_factory=deque)
    buf_lock: threading.Lock = field(default_factory=threading.Lock)

    delay_frames: int = 0
    delay_buf: Deque[np.ndarray] = field(default_factory=deque)

    rms: float = 0.0
    last_ts: float = 0.0

    dropped_in_frames: int = 0
    missing_out_frames: int = 0
    buffer_frames: int = 0
    src_rate: int = 0
