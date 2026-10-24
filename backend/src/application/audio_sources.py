from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from audio.domain.formats import AudioFormat
from audio.domain.ports import AudioSource

SourceErrorCallback = Callable[[str, str], None]


class AudioSourceFactory(Protocol):
    def create_loopback_source(
        self,
        *,
        name: str,
        engine_format: AudioFormat,
        device: Any,
        error_callback: SourceErrorCallback | None = None,
    ) -> AudioSource: ...

    def create_microphone_source(
        self, *, name: str, device: Any
    ) -> AudioSource: ...

    def create_process_source(
        self,
        *,
        name: str,
        token: Any,
        error_callback: SourceErrorCallback | None = None,
    ) -> AudioSource: ...
