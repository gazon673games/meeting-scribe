from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from audio.domain.formats import AudioFormat
from audio.infrastructure.sources.base import BaseSource

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import sounddevice as sd


class MicrophoneSource(BaseSource):
    def __init__(
        self, name: str, format: AudioFormat, device: int | None = None
    ):
        super().__init__(name=name, format=format)
        self._device = device
        self._stream: sd.InputStream | None = None
        self._on_audio: Callable[[str, np.ndarray], None] | None = None

    def start(self, on_audio: Callable[[str, np.ndarray], None]) -> None:
        import sounddevice as sd

        if self._stream is not None:
            raise RuntimeError('Microphone is already started')
        self._on_audio = on_audio

        self._stream = sd.InputStream(
            samplerate=self.format.sample_rate,
            channels=self.format.channels,
            dtype=self.format.dtype,
            blocksize=self.format.blocksize,
            device=self._device,
            callback=self._callback,
        )
        try:
            self._stream.start()
        except BaseException:
            # Keep ownership until close succeeds, including partial startup.
            try:
                self._stream.close()
            except Exception:
                _logger.exception(
                    'Cannot close microphone after failed startup'
                )
            else:
                self._stream = None
                self._on_audio = None
            raise

    def stop(self) -> None:
        if self._stream is not None:
            failure: BaseException | None = None
            try:
                self._stream.stop()
            except BaseException as error:
                failure = error
            try:
                self._stream.close()
            except BaseException:
                if failure is not None:
                    _logger.exception(
                        'Cannot close microphone after stop failure'
                    )
                    raise failure
                raise
            self._stream = None
            self._on_audio = None
            if failure is not None:
                raise failure

    def _callback(self, indata, frames, time_info, status):  # noqa: N803
        if self._on_audio is None:
            return
        # do not reshape/map here; engine will normalize
        x = np.asarray(indata)
        self._on_audio(self.name, x)
