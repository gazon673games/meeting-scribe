from __future__ import annotations

import unittest
from unittest.mock import Mock

from tools.audio.capture_cli import _finish_capture


class CaptureCleanupTests(unittest.TestCase):
    def test_failed_engine_stop_still_closes_writer_and_restores_handlers(self):
        stop = Mock(side_effect=OSError('stop'))
        close = Mock()
        restore = Mock()
        with (
            self.assertLogs('tools.audio.capture_cli', level='WARNING'),
            self.assertRaisesRegex(OSError, 'stop'),
        ):
            _finish_capture([stop, close, restore], active_error=False)
        close.assert_called_once()
        restore.assert_called_once()

    def test_cleanup_failure_preserves_active_processing_error(self):
        close = Mock(side_effect=OSError('close'))
        with self.assertLogs('tools.audio.capture_cli', level='WARNING'):
            _finish_capture([close], active_error=True)
