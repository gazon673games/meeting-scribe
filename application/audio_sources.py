from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

from audio.types import AudioFormat, AudioSource

SourceErrorCallback = Callable[[str, str], None]


class AudioSourceFactory(Protocol):
    def create_loopback_source(
        self,
        *,
        name: str,
        engine_format: AudioFormat,
        device: Any,
        error_callback: Optional[SourceErrorCallback] = None,
    ) -> AudioSource:
        ...

    def create_microphone_source(self, *, name: str, device: Any) -> AudioSource:
        ...
