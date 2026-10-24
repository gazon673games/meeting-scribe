from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Callable

import numpy as np

from audio.domain.formats import AudioFormat
from audio.infrastructure.sources.base import BaseSource

_logger = logging.getLogger(__name__)


class PulseAppSource(BaseSource):
    """Per-application audio capture via PulseAudio/PipeWire parec."""

    def __init__(self, name: str, sink_input_index: int):
        fmt = AudioFormat(
            sample_rate=48000, channels=2, dtype='float32', blocksize=1024
        )
        super().__init__(name=name, format=fmt)
        self._sink_input_index = int(sink_input_index)
        self._on_audio: Callable[[str, np.ndarray], None] | None = None
        self._on_error: Callable[[str, str], None] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._lifecycle_lock = threading.RLock()
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
                raise RuntimeError('PulseAudio capture is already started')
            if self._proc is not None:
                raise RuntimeError('Previous PulseAudio process needs cleanup')
            self._on_audio = on_audio
            self._set_last_error(None)
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name=f'{self.name}-pulse', daemon=True
            )
            try:
                self._thread.start()
            except BaseException:
                self._thread = None
                self._on_audio = None
                raise

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            proc = self._proc
            worker = self._thread
        if proc:
            try:
                proc.kill()
            except OSError:
                _logger.debug(
                    'Best-effort operation failed in stop', exc_info=True
                )
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=3.0)
            if worker.is_alive():
                raise TimeoutError('PulseAudio capture did not stop')
            with self._lifecycle_lock:
                if self._thread is worker:
                    self._thread = None

    def _run(self) -> None:
        try:
            cmd = [
                'parec',
                f'--monitor-stream={self._sink_input_index}',
                '--format=float32le',
                '--rate=48000',
                '--channels=2',
                '--latency-msec=50',
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            with self._lifecycle_lock:
                self._proc = proc
            chunk_frames = self.format.blocksize
            chunk_bytes = chunk_frames * 2 * 4  # 2 channels * 4 bytes float32
            pending = bytearray()

            while not self._stop.is_set():
                raw = proc.stdout.read(chunk_bytes - len(pending))
                if not raw:
                    break
                pending.extend(raw)
                if len(pending) < chunk_bytes:
                    continue
                arr = np.frombuffer(bytes(pending), dtype='<f4').reshape(-1, 2)
                pending.clear()
                if self._on_audio is not None:
                    self._on_audio(self.name, arr)
            complete_bytes = len(pending) - len(pending) % 8
            if (
                complete_bytes
                and self._on_audio is not None
                and not self._stop.is_set()
            ):
                tail = bytes(pending[:complete_bytes])
                self._on_audio(
                    self.name, np.frombuffer(tail, dtype='<f4').reshape(-1, 2)
                )

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
        finally:
            with self._lifecycle_lock:
                proc = self._proc
            if proc is not None:
                try:
                    if proc.poll() is None:
                        proc.kill()
                    proc.wait(timeout=2)
                    if proc.stdout is not None:
                        proc.stdout.close()
                except (OSError, subprocess.SubprocessError):
                    _logger.exception('Cannot close PulseAudio capture process')
                else:
                    with self._lifecycle_lock:
                        if self._proc is proc:
                            self._proc = None
