from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path

import numpy as np

from audio.domain.formats import AudioFormat

try:
    import soundfile as sf
except ImportError:
    sf = None


_logger = logging.getLogger(__name__)


def soundfile_available() -> bool:
    return sf is not None


class WavWriterThread(threading.Thread):
    def __init__(self, out_q: queue.Queue[np.ndarray]):
        super().__init__(name='wav-writer', daemon=True)
        self._frames = out_q
        self._stop_event = threading.Event()

        self._lock = threading.RLock()
        self._recording = False
        self._wav_file = None
        self._target_path: Path | None = None

        self._last_error: str | None = None
        self._written_blocks: int = 0
        self._drained_blocks: int = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                continue

            with self._lock:
                self._drained_blocks += 1
                if not self._recording or self._wav_file is None:
                    continue
                try:
                    self._wav_file.write(frame)
                    self._written_blocks += 1
                except Exception as error:
                    self._last_error = f'{type(error).__name__}: {error}'
                    self.stop_recording()

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=2.0)
        self.stop_recording()

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def written_blocks(self) -> int:
        with self._lock:
            return self._written_blocks

    def drained_blocks(self) -> int:
        with self._lock:
            return self._drained_blocks

    def start_recording(self, path: Path, fmt: AudioFormat) -> None:
        if sf is None:
            raise RuntimeError('soundfile is not installed')

        with self._lock:
            if self._recording:
                return

            self._last_error = None
            self._written_blocks = 0

            path.parent.mkdir(parents=True, exist_ok=True)
            wav_file = sf.SoundFile(
                str(path),
                mode='w',
                samplerate=fmt.sample_rate,
                channels=fmt.channels,
                subtype='PCM_16',
            )
            self._wav_file = wav_file
            self._target_path = path
            self._recording = True

    def stop_recording(self) -> None:
        with self._lock:
            self._recording = False
            if self._wav_file is not None:
                try:
                    self._wav_file.close()
                except Exception:
                    _logger.debug(
                        'Best-effort operation failed in stop_recording',
                        exc_info=True,
                    )
            self._wav_file = None
            self._target_path = None

    def target_path(self) -> Path | None:
        with self._lock:
            return self._target_path
