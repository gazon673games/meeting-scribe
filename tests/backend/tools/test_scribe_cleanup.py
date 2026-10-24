from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from tools.batch_scribe_parts.scribe import Scribe

from asr.infrastructure.offline_runner import OfflineProfile


class ScribeCleanupTests(unittest.TestCase):
    def test_failed_unload_retains_handle_for_retry(self):
        scribe = Scribe(OfflineProfile())
        model = Mock()
        model.model.unload_model.side_effect = [OSError('busy'), None]
        scribe._model = model
        with (
            self.assertLogs('tools.batch_scribe_parts.scribe', level='WARNING'),
            self.assertRaisesRegex(OSError, 'busy'),
        ):
            scribe.__exit__()
        self.assertIs(scribe._model, model)
        scribe.__exit__()
        self.assertIsNone(scribe._model)

    def test_cleanup_failure_does_not_replace_processing_failure(self):
        scribe = Scribe(OfflineProfile())
        scribe._model = Mock()
        scribe._model.model.unload_model.side_effect = OSError('busy')
        with (
            self.assertLogs('tools.batch_scribe_parts.scribe', level='WARNING'),
            self.assertRaisesRegex(ValueError, 'decode'),
            patch.object(Scribe, '__enter__', return_value=scribe),
        ):
            with scribe:
                raise ValueError('decode')
