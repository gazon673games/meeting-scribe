from __future__ import annotations

import unittest

import numpy as np

from asr.infrastructure.audio_utils import resample_linear


class ResamplingTests(unittest.TestCase):
    def test_48k_to_16k_preserves_speech_and_suppresses_aliasing(self) -> None:
        times = np.arange(48000, dtype=np.float64) / 48000.0
        levels = {}
        for frequency_hz in (1000, 6000, 12000):
            tone = np.sin(2 * np.pi * frequency_hz * times).astype(np.float32)
            output = resample_linear(tone, 48000, 16000)
            self.assertEqual(output.shape, (16000,))
            self.assertEqual(output.dtype, np.float32)
            steady = output[32:-32]
            levels[frequency_hz] = float(np.sqrt(np.mean(steady * steady)))

        self.assertGreater(levels[1000], 0.65)
        self.assertGreater(levels[6000], 0.5)
        self.assertLess(levels[12000], 0.02)


if __name__ == '__main__':
    unittest.main()
