from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, TypeAlias

from audio.domain.formats import AudioFormat

AudioFrame: TypeAlias = Any


class AudioFilter(Protocol):
    def process(self, frame: AudioFrame, fmt: AudioFormat) -> AudioFrame: ...


class AudioSource(Protocol):
    name: str

    def start(self, on_audio: Callable[[str, AudioFrame], None]) -> None: ...

    def stop(self) -> None: ...

    def get_format(self) -> AudioFormat: ...

    def get_filters(self) -> list[AudioFilter]: ...
