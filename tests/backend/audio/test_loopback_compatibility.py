from __future__ import annotations

import runpy
import sys
import unittest
import warnings
from unittest.mock import patch

import numpy

from audio.infrastructure.sources import wasapi_loopback as loopback


class _LegacyRecorder:
    def __init__(self, buffer: bytearray) -> None:
        self.buffer = buffer

    def _record_chunk(self) -> numpy.ndarray:
        return numpy.fromstring(self.buffer, dtype='float32')


class LoopbackCompatibilityTests(unittest.TestCase):
    def test_import_leaves_numpy_and_warning_filters_unchanged(self):
        fromstring = numpy.fromstring
        filters = list(warnings.filters)
        with patch.dict(sys.modules, {'soundcard': None}):
            runpy.run_path(loopback.__file__)
        self.assertIs(numpy.fromstring, fromstring)
        self.assertEqual(warnings.filters, filters)
        self.assertFalse(hasattr(numpy, '_meeting_scribe_fromstring_compat'))

    def test_decoder_is_local_and_owns_data_after_native_buffer_release(self):
        buffer = bytearray(numpy.array([0.25, -0.5], dtype='float32').tobytes())
        recorder = _LegacyRecorder(buffer)
        untouched = _LegacyRecorder(buffer)
        original_method = untouched._record_chunk.__func__
        fromstring = numpy.fromstring

        loopback._configure_recorder(recorder)
        data = recorder._record_chunk()
        buffer[:] = bytes(len(buffer))

        numpy.testing.assert_array_equal(data, [0.25, -0.5])
        self.assertTrue(data.flags.owndata)
        self.assertTrue(data.flags.writeable)
        self.assertIs(numpy.fromstring, fromstring)
        self.assertIs(untouched._record_chunk.__func__, original_method)
        self.assertIs(original_method.__globals__['numpy'], numpy)

    def test_adapter_preserves_text_parsing_and_binary_count(self):
        adapter = loopback._SoundcardNumpy()
        numpy.testing.assert_array_equal(
            adapter.fromstring('1,2,3', sep=',', count=2), [1, 2]
        )
        numpy.testing.assert_array_equal(
            adapter.fromstring(
                numpy.array([1, 2, 3], dtype='float32').tobytes(),
                dtype='float32',
                count=2,
            ),
            [1, 2],
        )
        with self.assertRaises(ValueError):
            adapter.fromstring(b'bad buffer', dtype='float32')

    def test_modern_recorder_keeps_its_original_decoder(self):
        class ModernRecorder:
            def _record_chunk(self):
                return numpy.frombuffer(b'', dtype='float32')

        recorder = ModernRecorder()
        original = recorder._record_chunk.__func__
        loopback._configure_recorder(recorder)
        self.assertIs(recorder._record_chunk.__func__, original)


if __name__ == '__main__':
    unittest.main()
