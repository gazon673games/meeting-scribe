from __future__ import annotations

import subprocess
import threading
from typing import Callable, Optional

import numpy as np

from audio.domain.formats import AudioFormat
from audio.infrastructure.sources.base import BaseSource


class PulseAppSource(BaseSource):
    """Per-application audio capture via PulseAudio/PipeWire parec."""

    def __init__(self, name: str, sink_input_index: int):
        fmt = AudioFormat(sample_rate=48000, channels=2, dtype="float32", blocksize=1024)
        super().__init__(name=name, format=fmt)
        self._sink_input_index = int(sink_input_index)
        self._on_audio: Optional[Callable[[str, np.ndarray], None]] = None
        self._on_error: Optional[Callable[[str, str], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._err_lock = threading.Lock()
        self._last_error: Optional[str] = None

    def set_error_callback(self, cb: Optional[Callable[[str, str], None]]) -> None:
        self._on_error = cb

    def get_last_error(self) -> Optional[str]:
        with self._err_lock:
            return self._last_error

    def _set_last_error(self, text: Optional[str]) -> None:
        with self._err_lock:
            self._last_error = text

    def start(self, on_audio: Callable[[str, np.ndarray], None]) -> None:
        self._on_audio = on_audio
        self._set_last_error(None)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-pulse", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc:
            try:
                proc.kill()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run(self) -> None:
        try:
            cmd = [
                "parec",
                f"--monitor-stream={self._sink_input_index}",
                "--format=float32le",
                "--rate=48000",
                "--channels=2",
                "--latency-msec=50",
            ]
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            chunk_frames = self.format.blocksize
            chunk_bytes = chunk_frames * 2 * 4  # 2 channels * 4 bytes float32

            while not self._stop.is_set():
                raw = self._proc.stdout.read(chunk_bytes)
                if not raw:
                    break
                if len(raw) < chunk_bytes:
                    continue
                arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)
                if self._on_audio is not None:
                    self._on_audio(self.name, arr)

        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            self._set_last_error(err)
            if self._on_error:
                try:
                    self._on_error(self.name, err)
                except Exception:
                    pass
        finally:
            proc = self._proc
            self._proc = None
            if proc:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
