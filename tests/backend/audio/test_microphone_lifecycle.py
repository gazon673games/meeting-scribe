from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from audio.domain.formats import AudioFormat
from audio.infrastructure.sources.microphone import MicrophoneSource


class MicrophoneLifecycleTests(unittest.TestCase):
    def test_failed_start_closes_partial_stream_and_allows_retry(self):
        stream = Mock()
        stream.start.side_effect = OSError('device unavailable')
        sdk = Mock(InputStream=Mock(return_value=stream))
        source = MicrophoneSource('mic', AudioFormat(48000, 1))
        with patch.dict('sys.modules', {'sounddevice': sdk}):
            with self.assertRaisesRegex(OSError, 'device unavailable'):
                source.start(Mock())
            stream.close.assert_called_once()
            self.assertIsNone(source._stream)
            stream.start.side_effect = None
            source.start(Mock())
            source.stop()
        self.assertIsNone(source._stream)

    def test_failed_stop_and_close_preserve_primary_error_and_owned_handle(
        self,
    ):
        stream = Mock()
        primary = OSError('stop failed')
        stream.stop.side_effect = primary
        stream.close.side_effect = OSError('close failed')
        source = MicrophoneSource('mic', AudioFormat(48000, 1))
        source._stream = stream
        with (
            self.assertLogs(
                'audio.infrastructure.sources.microphone', level='ERROR'
            ),
            self.assertRaises(OSError) as raised,
        ):
            source.stop()
        self.assertIs(raised.exception, primary)
        self.assertIs(source._stream, stream)
        stream.stop.side_effect = None
        stream.close.side_effect = None
        source.stop()
        self.assertIsNone(source._stream)
