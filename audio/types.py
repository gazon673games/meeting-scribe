from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Protocol

import numpy as np


@dataclass(frozen=True)
class AudioFormat:
    sample_rate: int
    channels: int
    dtype: str = "float32"
    blocksize: int = 1024


class AudioFilter(Protocol):
    def process(self, x: np.ndarray, fmt: AudioFormat) -> np.ndarray:
        ...


class AudioSource(Protocol):
    name: str

    def start(self, on_audio: Callable[[str, np.ndarray], None]) -> None:
        ...

    def stop(self) -> None:
        ...

    def get_format(self) -> AudioFormat:
        ...

    def get_filters(self) -> List[AudioFilter]:
        ...


TapMode = Literal["mix", "sources", "both"]
