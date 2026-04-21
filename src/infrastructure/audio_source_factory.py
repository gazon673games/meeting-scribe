from __future__ import annotations

from typing import Any, Optional

from application.audio_sources import AudioSourceFactory, SourceErrorCallback
from audio.domain.formats import AudioFormat
from audio.domain.ports import AudioSource


class DefaultAudioSourceFactory(AudioSourceFactory):
    def create_loopback_source(
        self,
        *,
        name: str,
        engine_format: AudioFormat,
        device: Any,
        error_callback: Optional[SourceErrorCallback] = None,
    ) -> AudioSource:
        from audio.infrastructure.sources.wasapi_loopback import WasapiLoopbackSource

        source = WasapiLoopbackSource(name=name, format=engine_format, device=device)
        if error_callback is not None:
            source.set_error_callback(error_callback)
        return source

    def create_microphone_source(self, *, name: str, device: Any) -> AudioSource:
        from audio.infrastructure.sources.microphone import MicrophoneSource

        mic_format = AudioFormat(sample_rate=48000, channels=1, dtype="float32", blocksize=1024)
        return MicrophoneSource(name=name, format=mic_format, device=int(device))
