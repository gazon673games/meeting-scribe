from __future__ import annotations

import io
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from audio.infrastructure.sources.pulse_app_source import PulseAppSource


class _ShortReads(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 7))


class PulseCaptureTests(unittest.TestCase):
    def test_short_pipe_reads_preserve_samples_and_final_aligned_tail(self):
        samples = np.arange(2050, dtype='<f4').reshape(-1, 2)
        process = Mock(stdout=_ShortReads(samples.tobytes()))
        process.poll.return_value = 0
        source = PulseAppSource('application', 1)
        frames = []
        source._on_audio = lambda name, audio: frames.append(audio)
        with patch(
            'audio.infrastructure.sources.pulse_app_source.subprocess.Popen',
            return_value=process,
        ):
            source._run()
        self.assertEqual([len(frame) for frame in frames], [1024, 1])
        np.testing.assert_array_equal(np.concatenate(frames), samples)
        self.assertTrue(process.stdout.closed)
        self.assertIsNone(source._proc)

    def test_stop_during_process_creation_prevents_blocking_read(self):
        entered = threading.Event()
        release = threading.Event()
        process = Mock()
        process.poll.return_value = None
        source = PulseAppSource('application', 1)

        def create(*args, **kwargs):
            entered.set()
            if not release.wait(3):
                raise TimeoutError('Test did not release process creation')
            return process

        with patch(
            'audio.infrastructure.sources.pulse_app_source.subprocess.Popen',
            side_effect=create,
        ):
            source.start(Mock())
            try:
                self.assertTrue(entered.wait(3))
                source._stop.set()
            finally:
                release.set()
                source.stop()
        process.stdout.read.assert_not_called()
        process.wait.assert_called_once()
        self.assertIsNone(source._thread)
