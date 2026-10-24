from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import FunctionType, MethodType
from typing import Any, Protocol

import numpy as np

from audio.domain.formats import AudioFormat
from audio.infrastructure.sources.base import BaseSource

_logger = logging.getLogger(__name__)


class _RecorderPort(Protocol):
    def record(self, numframes: int) -> np.ndarray | None: ...


class _MicrophonePort(Protocol):
    id: str
    name: str
    isloopback: bool

    def recorder(
        self, *, samplerate: int, blocksize: int
    ) -> AbstractContextManager[_RecorderPort]: ...


class _SoundcardNumpy:
    """Keep SoundCard's binary decoder local to one recorder method."""

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)

    @staticmethod
    def fromstring(
        data: Any,
        dtype: Any = float,
        count: int = -1,
        sep: str = '',
        **kwargs: Any,
    ) -> np.ndarray:
        if sep:
            return np.fromstring(
                data, dtype=dtype, count=count, sep=sep, **kwargs
            )
        # WASAPI releases the native buffer before returning the chunk.
        return np.frombuffer(data, dtype=dtype, count=count, **kwargs).copy()


def _configure_recorder(recorder: Any) -> None:
    # SoundCard 0.4.x calls the removed binary numpy.fromstring API.
    # Rebind only this instance's method; leave NumPy and SDK globals intact.
    method = getattr(recorder, '_record_chunk', None)
    if not isinstance(method, MethodType):
        return
    function = method.__func__
    if (
        function.__globals__.get('numpy') is not np
        or 'fromstring' not in function.__code__.co_names
    ):
        return
    local_function = FunctionType(
        function.__code__,
        {**function.__globals__, 'numpy': _SoundcardNumpy()},
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    local_function.__kwdefaults__ = function.__kwdefaults__
    recorder._record_chunk = MethodType(local_function, recorder)


def _pick_default_loopback(
    soundcard: Any, devices: list[_MicrophonePort]
) -> _MicrophonePort:
    speaker = soundcard.default_speaker()
    for microphone in devices:
        if microphone.id == speaker.id:
            return microphone
    return soundcard.get_microphone(speaker.name, include_loopback=True)


def _find_device(
    soundcard: Any, devices: list[_MicrophonePort], device: str | int
) -> _MicrophonePort:
    device_name = str(device).strip().lower()
    for microphone in devices:
        if microphone.id == device:
            return microphone
    for microphone in devices:
        if microphone.name.strip().lower() == device_name:
            return microphone
    for microphone in devices:
        if device_name and device_name in microphone.name.strip().lower():
            return microphone
    return soundcard.get_microphone(str(device), include_loopback=True)


class WasapiLoopbackSource(BaseSource):
    """WASAPI loopback source implemented via SoundCard."""

    def __init__(
        self, name: str, format: AudioFormat, device: str | int | None = None
    ) -> None:
        super().__init__(name=name, format=format)
        self._device = device
        self._on_audio: Callable[[str, np.ndarray], None] | None = None
        self._on_error: Callable[[str, str], None] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lifecycle_lock = threading.Lock()
        self._err_lock = threading.Lock()
        self._last_error: str | None = None

    def set_error_callback(self, cb: Callable[[str, str], None] | None) -> None:
        self._on_error = cb

    def get_last_error(self) -> str | None:
        with self._err_lock:
            return self._last_error

    def _set_last_error(self, text: str | None) -> None:
        with self._err_lock:
            self._last_error = text

    def start(self, on_audio: Callable[[str, np.ndarray], None]) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                if self._stop.is_set():
                    raise RuntimeError('Previous loopback worker is stopping')
                return
            self._on_audio = on_audio
            self._set_last_error(None)
            self._stop = threading.Event()
            worker = threading.Thread(
                target=self._run,
                name=f'{self.name}-loopback',
                daemon=True,
            )
            self._thread = worker
            try:
                worker.start()
            except BaseException:
                self._thread = None
                raise

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            worker = self._thread
        if worker is None or worker is threading.current_thread():
            return
        worker.join(timeout=2.0)
        with self._lifecycle_lock:
            if self._thread is worker and not worker.is_alive():
                self._thread = None

    def _pick_loopback_mic(self) -> _MicrophonePort:
        import soundcard

        candidates = list(soundcard.all_microphones(include_loopback=True))
        loopbacks = [device for device in candidates if device.isloopback]
        devices = loopbacks if loopbacks else candidates
        if not devices:
            raise RuntimeError('No capture devices were found')
        if self._device is None:
            return _pick_default_loopback(soundcard, devices)
        return _find_device(soundcard, devices, self._device)

    def _run(self) -> None:
        stop = self._stop
        on_audio = self._on_audio
        try:
            microphone = self._pick_loopback_mic()
            recorder = microphone.recorder(
                samplerate=self.format.sample_rate,
                blocksize=self.format.blocksize,
            )
            _configure_recorder(recorder)
            with recorder as recording:
                while not stop.is_set():
                    data = recording.record(numframes=self.format.blocksize)
                    if data is not None and on_audio is not None:
                        on_audio(self.name, np.asarray(data, dtype=np.float32))
        except Exception as error:
            message = f'{type(error).__name__}: {error}'
            self._set_last_error(message)
            _logger.exception('Loopback capture failed for %s', self.name)
            callback = self._on_error
            if callback is not None:
                try:
                    callback(self.name, message)
                except Exception:
                    _logger.exception(
                        'Loopback error callback failed for %s', self.name
                    )
