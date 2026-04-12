from __future__ import annotations

from typing import Any, Callable, Optional

from audio.types import AudioFormat, AudioSource
from audio.sources.microphone import MicrophoneSource
from audio.sources.wasapi_loopback import WasapiLoopbackSource

SourceErrorCallback = Callable[[str, str], None]


def create_loopback_source(
    *,
    name: str,
    engine_format: AudioFormat,
    device: Any,
    error_callback: Optional[SourceErrorCallback] = None,
) -> AudioSource:
    source = WasapiLoopbackSource(name=name, format=engine_format, device=device)
    if error_callback is not None:
        source.set_error_callback(error_callback)
    return source


def create_microphone_source(*, name: str, device: Any) -> AudioSource:
    mic_format = AudioFormat(sample_rate=48000, channels=1, dtype="float32", blocksize=1024)
    return MicrophoneSource(name=name, format=mic_format, device=int(device))
