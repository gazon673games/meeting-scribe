from __future__ import annotations

import threading
import unittest

import numpy as np

from asr.infrastructure.audio_data import MonoAudio16kBuffer
from asr.infrastructure.audio_utils import resample_linear, stereo_to_mono
from asr.infrastructure.gain import PreGainAGC
from asr.infrastructure.runtime_workers import ThreadRealtimeWorkerRunner


class ASRAudioInfrastructureHelperTests(unittest.TestCase):
    def test_mono_buffer_reports_samples_frames_and_duration(self) -> None:
        buffer = MonoAudio16kBuffer.from_array([0.0, 0.5, -0.5, 1.0], sample_rate_hz=4)

        self.assertEqual(buffer.frame_count, 4)
        self.assertEqual(buffer.duration_s, 1.0)
        self.assertTrue(np.array_equal(buffer.samples, np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)))

    def test_audio_utils_convert_channels_and_resample_common_paths(self) -> None:
        mono = np.array([1.0, -1.0], dtype=np.float32)
        self.assertIs(stereo_to_mono(mono), mono)
        self.assertTrue(np.array_equal(stereo_to_mono(np.array([[1.0], [2.0]], dtype=np.float32)), [1.0, 2.0]))

        stereo = np.array([[1.0, 0.1], [1.0, 0.1]], dtype=np.float32)
        self.assertTrue(np.allclose(stereo_to_mono(stereo), [1.0, 1.0]))
        balanced = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
        self.assertTrue(np.allclose(stereo_to_mono(balanced), [0.0, 0.0]))

        same = resample_linear(mono, 16_000, 16_000)
        self.assertTrue(np.array_equal(same, mono))
        self.assertEqual(resample_linear(np.array([1.0], dtype=np.float32), 48_000, 16_000).shape[0], 1)
        self.assertEqual(resample_linear(np.ones(48, dtype=np.float32), 48_000, 16_000).shape[0], 16)
        self.assertEqual(resample_linear(np.ones(10, dtype=np.float32), 10, 20).shape[0], 20)

    def test_pre_gain_agc_tracks_input_rms_and_clips_output(self) -> None:
        agc = PreGainAGC(target_rms=0.5, max_gain=2.0, alpha=1.0)

        self.assertEqual(agc.process(np.array([], dtype=np.float32)).shape[0], 0)
        amplified = agc.process(np.array([0.25, -0.25], dtype=np.float32))

        self.assertGreater(agc.last_in_rms, 0.0)
        self.assertTrue(np.all(np.abs(amplified) <= 1.0))
        self.assertAlmostEqual(PreGainAGC._rms(np.array([3.0, 4.0], dtype=np.float32)), 3.5355339, places=5)

    def test_thread_worker_runner_creates_stop_signal_and_worker_handle(self) -> None:
        runner = ThreadRealtimeWorkerRunner()
        stop_signal = runner.create_stop_signal()
        done = threading.Event()

        handle = runner.start_worker(name="unit-worker", target=done.set)
        handle.join(timeout=1.0)

        self.assertFalse(stop_signal.is_set())
        self.assertTrue(done.is_set())


if __name__ == "__main__":
    unittest.main()
