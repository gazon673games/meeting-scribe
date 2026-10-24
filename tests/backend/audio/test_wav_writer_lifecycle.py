from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from audio.domain.formats import AudioFormat
from audio.infrastructure.writer import WavWriterThread


class WavWriterLifecycleTests(unittest.TestCase):
    def test_real_thread_writes_then_joins_and_closes_recording(self) -> None:
        written = threading.Event()
        wav_file = Mock()
        wav_file.write.side_effect = lambda frame: written.set()
        frames: queue.Queue[np.ndarray] = queue.Queue()
        writer = WavWriterThread(frames)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch('audio.infrastructure.writer.sf') as soundfile,
        ):
            soundfile.SoundFile.return_value = wav_file
            writer.start_recording(
                Path(directory) / 'recording.wav',
                AudioFormat(sample_rate=16_000, channels=1),
            )
            writer.start()
            try:
                frames.put(np.zeros((2, 1), dtype=np.float32))
                self.assertTrue(written.wait(timeout=2))
            finally:
                writer.stop()
        self.assertFalse(writer.is_alive())
        self.assertEqual(writer.written_blocks(), 1)
        self.assertFalse(writer.is_recording())
        wav_file.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
