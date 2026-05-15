from __future__ import annotations

import queue
import unittest

import numpy as np

from audio.application.dsp import (
    apply_delay_block,
    apply_filters,
    channel_map_to_engine,
    pad_or_crop_n,
    resample_to_engine_rate,
    rms,
)
from audio.application.source_state import SourceState
from audio.application.tap import build_tap_packet, tap_should_send, try_emit_tap_packet
from audio.domain.formats import AudioFormat


class _ScaleFilter:
    def __init__(self, factor: float) -> None:
        self.factor = factor

    def process(self, x: np.ndarray, fmt: AudioFormat) -> np.ndarray:
        return x * self.factor


class _BadSizeQueue(queue.Queue):
    def qsize(self) -> int:
        raise RuntimeError("unknown")


class AudioDspAndTapTests(unittest.TestCase):
    def test_filters_rms_and_padding_transform_audio_blocks(self) -> None:
        fmt = AudioFormat(sample_rate=48_000, channels=2)
        block = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)

        filtered = apply_filters(block, fmt, [_ScaleFilter(2.0), _ScaleFilter(0.5)])

        np.testing.assert_allclose(filtered, block)
        self.assertEqual(rms(np.array([], dtype=np.float32)), 0.0)
        self.assertAlmostEqual(rms(block), float(np.sqrt(0.625)))
        self.assertEqual(pad_or_crop_n(block, 1).shape, (1, 2))
        padded = pad_or_crop_n(block, 4)
        self.assertEqual(padded.shape, (4, 2))
        np.testing.assert_allclose(padded[2:], 0.0)

    def test_channel_mapping_and_resampling_preserve_expected_shapes(self) -> None:
        mono = np.array([[1.0], [3.0]], dtype=np.float32)
        stereo = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)
        tri = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        np.testing.assert_allclose(channel_map_to_engine(mono, 1, 2), [[1.0, 1.0], [3.0, 3.0]])
        np.testing.assert_allclose(channel_map_to_engine(stereo, 2, 1), [[2.0], [6.0]])
        np.testing.assert_allclose(channel_map_to_engine(tri, 3, 2), [[1.0, 2.0]])
        np.testing.assert_allclose(channel_map_to_engine(mono, 1, 3), [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

        same_rate = resample_to_engine_rate(stereo, 48_000, 48_000)
        self.assertEqual(same_rate.shape, stereo.shape)
        self.assertEqual(resample_to_engine_rate(stereo[:0], 48_000, 16_000).shape, (0, 2))
        self.assertEqual(resample_to_engine_rate(stereo, 2, 4).shape, (4, 2))

    def test_delay_block_returns_zeroes_until_delayed_block_is_due(self) -> None:
        state = SourceState(src=object())  # type: ignore[arg-type]
        state.delay_frames = 960
        block = np.ones((480, 2), dtype=np.float32)

        first = apply_delay_block(state, block, 480, 2)
        second = apply_delay_block(state, block * 2, 480, 2)
        third = apply_delay_block(state, block * 3, 480, 2)
        fourth = apply_delay_block(state, block * 4, 480, 2)

        np.testing.assert_allclose(first, 0.0)
        np.testing.assert_allclose(second, 0.0)
        np.testing.assert_allclose(third, block)
        np.testing.assert_allclose(fourth, block * 2)

    def test_tap_packets_copy_selected_audio_and_drop_when_queue_is_busy(self) -> None:
        mixed = np.array([[1.0]], dtype=np.float32)
        source_block = np.array([[2.0]], dtype=np.float32)

        packet = build_tap_packet(
            t_start=1.25,
            t_end=1.5,
            mixed=mixed,
            sources_out={"mic": source_block},
            mode="both",
        )
        mixed[0, 0] = 9.0
        source_block[0, 0] = 8.0

        self.assertEqual(packet["t_start"], 1.25)
        np.testing.assert_allclose(packet["mix"], [[1.0]])
        np.testing.assert_allclose(packet["sources"]["mic"], [[2.0]])  # type: ignore[index]

        tap_q: queue.Queue[dict] = queue.Queue(maxsize=1)
        tap_q.put_nowait({"already": "full"})
        self.assertFalse(tap_should_send(tap_q, tap_queue_max=1, drop_threshold=0.5))
        self.assertTrue(tap_should_send(_BadSizeQueue(), tap_queue_max=1, drop_threshold=0.5))
        self.assertFalse(
            try_emit_tap_packet(
                tap_q=tap_q,
                tap_queue_max=10,
                drop_threshold=1.0,
                t_start=0.0,
                t_end=0.1,
                mixed=np.zeros((1, 1), dtype=np.float32),
                sources_out=None,
                mode="mix",
            )
        )


if __name__ == "__main__":
    unittest.main()
