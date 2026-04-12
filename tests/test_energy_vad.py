from __future__ import annotations

import unittest

import numpy as np

from asr.vad import EnergyVAD


def _sine_frame(freq_hz: float, amplitude: float, sample_rate: int, frame_len: int) -> np.ndarray:
    t = np.arange(frame_len, dtype=np.float32) / float(sample_rate)
    return (float(amplitude) * np.sin(2.0 * np.pi * float(freq_hz) * t)).astype(np.float32)


class EnergyVADTests(unittest.TestCase):
    def test_strong_voiced_speech_frame_is_detected(self) -> None:
        vad = EnergyVAD(
            sample_rate=16000,
            frame_ms=20,
            energy_threshold=0.01,
            adaptive=False,
            voiced_every_n_frames=1,
        )

        frame = _sine_frame(freq_hz=350.0, amplitude=0.06, sample_rate=vad.sample_rate, frame_len=vad.frame_len)

        self.assertTrue(vad.is_speech_frame(frame))
        self.assertGreater(vad.last_rms(), vad.last_threshold())
        self.assertGreater(vad.last_band_ratio(), 0.35)
        self.assertGreater(vad.last_voiced(), 0.12)

    def test_near_threshold_speech_like_frame_uses_band_and_voiced_path(self) -> None:
        vad = EnergyVAD(
            sample_rate=16000,
            frame_ms=20,
            energy_threshold=0.02,
            adaptive=False,
            voiced_every_n_frames=1,
        )

        frame = _sine_frame(freq_hz=350.0, amplitude=0.027, sample_rate=vad.sample_rate, frame_len=vad.frame_len)

        self.assertTrue(vad.is_speech_frame(frame))
        self.assertLess(vad.last_rms(), vad.last_threshold())
        self.assertGreaterEqual(vad.last_rms(), 0.85 * vad.last_threshold())
        self.assertGreater(vad.last_band_ratio(), 0.35)
        self.assertGreater(vad.last_voiced(), 0.12)

    def test_adaptive_threshold_increases_after_repeated_non_speech_noise(self) -> None:
        vad = EnergyVAD(
            sample_rate=16000,
            frame_ms=20,
            energy_threshold=0.006,
            adaptive=True,
            noise_alpha=0.5,
            voiced_every_n_frames=1,
        )

        noise_frame = _sine_frame(freq_hz=6000.0, amplitude=0.004, sample_rate=vad.sample_rate, frame_len=vad.frame_len)

        decisions = [vad.is_speech_frame(noise_frame) for _ in range(12)]

        self.assertTrue(all(decision is False for decision in decisions))
        self.assertGreater(vad.noise_rms(), 0.0)
        self.assertGreater(vad.last_threshold(), 0.007)

    def test_hangover_keeps_short_silence_marked_as_speech(self) -> None:
        vad = EnergyVAD(
            sample_rate=16000,
            frame_ms=20,
            energy_threshold=0.01,
            adaptive=False,
            hangover_ms=40,
            min_end_silence_ms=0,
            voiced_every_n_frames=1,
        )

        speech = _sine_frame(freq_hz=350.0, amplitude=0.06, sample_rate=vad.sample_rate, frame_len=vad.frame_len)
        silence = np.zeros((vad.frame_len,), dtype=np.float32)

        results = [
            vad.is_speech_frame(speech),
            vad.is_speech_frame(silence),
            vad.is_speech_frame(silence),
            vad.is_speech_frame(silence),
        ]

        self.assertEqual(results, [True, True, True, False])

    def test_min_end_silence_keeps_utterance_open_briefly(self) -> None:
        vad = EnergyVAD(
            sample_rate=16000,
            frame_ms=20,
            energy_threshold=0.01,
            adaptive=False,
            hangover_ms=0,
            min_end_silence_ms=40,
            voiced_every_n_frames=1,
        )

        speech = _sine_frame(freq_hz=350.0, amplitude=0.06, sample_rate=vad.sample_rate, frame_len=vad.frame_len)
        silence = np.zeros((vad.frame_len,), dtype=np.float32)

        results = [
            vad.is_speech_frame(speech),
            vad.is_speech_frame(silence),
            vad.is_speech_frame(silence),
            vad.is_speech_frame(silence),
        ]

        self.assertEqual(results, [True, True, True, False])

    def test_preroll_returns_recent_frames_and_then_clears(self) -> None:
        vad = EnergyVAD(
            sample_rate=16000,
            frame_ms=20,
            energy_threshold=0.1,
            adaptive=False,
            pre_speech_ms=60,
            voiced_every_n_frames=1,
        )

        frames = [
            _sine_frame(freq_hz=50.0, amplitude=amp, sample_rate=vad.sample_rate, frame_len=vad.frame_len)
            for amp in (0.002, 0.003, 0.004, 0.005)
        ]

        for frame in frames:
            self.assertFalse(vad.is_speech_frame(frame))

        preroll, count = vad.pop_preroll()

        self.assertEqual(count, 3)
        self.assertEqual(preroll.shape, (vad.frame_len * 3,))
        np.testing.assert_allclose(preroll[: vad.frame_len], frames[1])
        np.testing.assert_allclose(preroll[vad.frame_len : 2 * vad.frame_len], frames[2])
        np.testing.assert_allclose(preroll[2 * vad.frame_len :], frames[3])

        empty, empty_count = vad.pop_preroll()
        self.assertEqual(empty_count, 0)
        self.assertEqual(empty.shape, (0,))


if __name__ == "__main__":
    unittest.main()
