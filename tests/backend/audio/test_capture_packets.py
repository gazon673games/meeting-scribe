from __future__ import annotations

import ctypes
import unittest
from unittest.mock import patch

import numpy as np

from audio.infrastructure.sources import process_loopback_win as capture
from audio.infrastructure.sources.wasapi_bindings import WAVEFORMATEX


class CapturePacketTests(unittest.TestCase):
    def setUp(self):
        self.source = capture.ProcessLoopbackWinSource('process', 123)
        self.format = WAVEFORMATEX()
        self.format.wFormatTag = 1
        self.format.wBitsPerSample = 16
        self.format.nChannels = 2
        self.format.nBlockAlign = 4
        self.calls = []

    def call_native(self, pointer, index, result_type, argument_types, *args):
        self.calls.append(index)
        if index == 5:
            args[0]._obj.value = 1
        elif index == 3:
            args[1]._obj.value = 1
            args[2]._obj.value = 2  # Silent packets may have a null pointer.
        return 0

    def test_silent_packet_is_released_before_delivering_owned_samples(self):
        frames = []

        def on_audio(name, frame):
            self.assertEqual(self.calls, [5, 3, 4])
            frames.append(frame)
            self.source._stop.set()

        self.source._on_audio = on_audio
        with patch.object(capture, '_wincall', side_effect=self.call_native):
            self.source._capture_loop(None, self.format)
        np.testing.assert_array_equal(frames[0], np.zeros((1, 2), np.float32))

    def test_decode_failure_still_releases_packet(self):
        with (
            patch.object(capture, '_wincall', side_effect=self.call_native),
            patch.object(
                capture, '_decode_packet', side_effect=ValueError('decode')
            ),
            self.assertRaisesRegex(ValueError, 'decode'),
        ):
            self.source._capture_loop(None, self.format)
        self.assertEqual(self.calls, [5, 3, 4])

    def test_pcm16_conversion_preserves_signed_channels(self):
        raw = np.array([-32768, 32767, 0, -16384], dtype='<i2')
        buffer = ctypes.create_string_buffer(raw.tobytes())
        result = capture._decode_packet(
            ctypes.cast(buffer, ctypes.c_void_p), 2, 0, self.format
        )
        np.testing.assert_array_equal(
            result, raw.astype(np.float32).reshape(2, 2) / 32768.0
        )
