from __future__ import annotations

from typing import Any, Optional

from application.audio_runtime import AudioRuntimeFactory, AudioRuntimePort
from audio.application.engine import AudioEngine
from audio.domain.formats import AudioFormat


class DefaultAudioRuntimeFactory(AudioRuntimeFactory):
    def create(
        self,
        *,
        format: AudioFormat,
        output_queue: Any,
        tap_queue: Optional[Any] = None,
    ) -> AudioRuntimePort:
        return AudioEngine(format=format, output_queue=output_queue, tap_queue=tap_queue)
