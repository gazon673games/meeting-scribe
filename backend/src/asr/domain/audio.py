from __future__ import annotations

from typing import Protocol


class MonoAudio16k(Protocol):
    @property
    def sample_rate_hz(self) -> int:
        ...

    @property
    def frame_count(self) -> int:
        ...

    @property
    def duration_s(self) -> float:
        ...

    @property
    def samples(self) -> object:
        ...
