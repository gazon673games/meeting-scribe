from __future__ import annotations

import ctypes
import logging
import sys
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from audio.domain.formats import AudioFormat
from audio.infrastructure.sources.base import BaseSource
from audio.infrastructure.sources.wasapi_bindings import (
    S_OK,
    WAVE_FORMAT_IEEE_FLOAT,
    WAVE_FORMAT_PCM,
    WAVEFORMATEX,
    _wincall,
    capture_process_loopback,
)

_logger = logging.getLogger(__name__)


class ProcessLoopbackWinSource(BaseSource):
    """Per-process audio capture using WASAPI Process Loopback (Windows 10 2004+)."""

    def __init__(self, name: str, pid: int):
        fmt = AudioFormat(
            sample_rate=44100, channels=2, dtype='float32', blocksize=1024
        )
        super().__init__(name=name, format=fmt)
        self.pid = int(pid)
        self._on_audio: Callable[[str, np.ndarray], None] | None = None
        self._on_error: Callable[[str, str], None] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
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
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError('Process loopback capture is already started')
        self._on_audio = on_audio
        self._set_last_error(None)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f'{self.name}-procloop', daemon=True
        )
        try:
            self._thread.start()
        except BaseException:
            self._thread = None
            self._on_audio = None
            raise

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            if self._thread is threading.current_thread():
                return
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                raise TimeoutError('Process loopback capture did not stop')
            self._thread = None

    def _run(self) -> None:
        try:
            capture_process_loopback(self.pid, self._capture_loop)
        except Exception as error:
            err = f'{type(error).__name__}: {error}'
            self._set_last_error(err)
            if self._on_error:
                try:
                    self._on_error(self.name, err)
                except Exception:
                    _logger.debug(
                        'Best-effort operation failed in _run', exc_info=True
                    )

    def _capture_loop(self, p_capture: Any, wfx: WAVEFORMATEX) -> None:
        while not self._stop.is_set():
            packet_frames = ctypes.c_uint32(0)
            hr = _wincall(
                p_capture,
                5,
                ctypes.HRESULT,
                [ctypes.POINTER(ctypes.c_uint32)],
                ctypes.byref(packet_frames),
            )
            if hr != S_OK:
                raise OSError(
                    f'GetNextPacketSize failed: 0x{hr & 0xFFFFFFFF:08x}'
                )
            if packet_frames.value == 0:
                self._stop.wait(0.005)
                continue
            # IAudioCaptureClient::GetBuffer(&pData, &numFrames, &flags, NULL, NULL) → vtable[3]
            p_data = ctypes.c_void_p()
            num_frames = ctypes.c_uint32(0)
            flags = ctypes.c_uint32(0)
            hr = _wincall(
                p_capture,
                3,
                ctypes.HRESULT,
                [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                ],
                ctypes.byref(p_data),
                ctypes.byref(num_frames),
                ctypes.byref(flags),
                None,
                None,
            )

            if hr < 0:
                raise OSError(f'GetBuffer failed: 0x{hr & 0xFFFFFFFF:08x}')
            if num_frames.value == 0:
                self._stop.wait(0.005)
                continue

            n = num_frames.value
            try:
                arr = _decode_packet(p_data, n, flags.value, wfx)
            finally:
                # The native lease ends before an application callback runs.
                release_hr = _wincall(
                    p_capture, 4, ctypes.HRESULT, [ctypes.c_uint32], n
                )
                if release_hr != S_OK:
                    message = (
                        f'ReleaseBuffer failed: 0x{release_hr & 0xFFFFFFFF:08x}'
                    )
                    if sys.exc_info()[0] is None:
                        raise OSError(message)
                    _logger.error(message)

            if self._on_audio is not None and arr.shape[0] > 0:
                self._on_audio(self.name, arr)


def _decode_packet(
    data: ctypes.c_void_p, frames: int, flags: int, format: WAVEFORMATEX
) -> np.ndarray:
    if flags & 0x2:  # AUDCLNT_BUFFERFLAGS_SILENT permits a null data pointer.
        return np.zeros((frames, format.nChannels), dtype=np.float32)
    if not data.value:
        raise ValueError('Capture returned a null pointer for non-silent audio')
    raw = ctypes.string_at(data, frames * format.nBlockAlign)
    if format.wFormatTag == WAVE_FORMAT_PCM and format.wBitsPerSample == 16:
        samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    elif (
        format.wFormatTag == WAVE_FORMAT_IEEE_FLOAT
        and format.wBitsPerSample == 32
    ):
        samples = np.frombuffer(raw, dtype='<f4').copy()
    else:
        raise ValueError(
            f'Unsupported capture format: {format.wFormatTag}/{format.wBitsPerSample}'
        )
    return samples.reshape(frames, format.nChannels)
