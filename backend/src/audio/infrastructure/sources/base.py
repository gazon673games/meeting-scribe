from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from audio.domain.formats import AudioFormat
from audio.domain.ports import AudioFilter


@dataclass
class BaseSource:
    name: str
    format: AudioFormat
    _filters: List[AudioFilter] = field(default_factory=list, init=False)

    def add_filter(self, flt: AudioFilter) -> None:
        self._filters.append(flt)

    def get_filters(self) -> List[AudioFilter]:
        return list(self._filters)

    def get_format(self) -> AudioFormat:
        return self.format

    # start/stop реализуются в конкретных источниках
