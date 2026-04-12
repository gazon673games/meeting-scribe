from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import sounddevice as sd

from audio.domain import AudioFormat
from audio.infrastructure.sources.base import BaseSource


class MicrophoneSource(BaseSource):
    def __init__(self, name: str, format: AudioFormat, device: Optional[int] = None):
        super().__init__(name=name, format=format)
        self._device = device
        self._stream: Optional[sd.InputStream] = None
        self._on_audio: Optional[Callable[[str, np.ndarray], None]] = None

    def start(self, on_audio: Callable[[str, np.ndarray], None]) -> None:
        self._on_audio = on_audio

        self._stream = sd.InputStream(
            samplerate=self.format.sample_rate,
            channels=self.format.channels,
            dtype=self.format.dtype,
            blocksize=self.format.blocksize,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None

    def _callback(self, indata, frames, time_info, status):  # noqa: N803
        if self._on_audio is None:
            return
        # do not reshape/map here; engine will normalize
        x = np.asarray(indata)
        self._on_audio(self.name, x)
