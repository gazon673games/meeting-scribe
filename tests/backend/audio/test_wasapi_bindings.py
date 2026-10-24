from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

from audio.infrastructure.sources import wasapi_bindings as bindings


class WasapiCompletionTests(unittest.TestCase):
    def test_completion_keeps_callbacks_alive_and_exposes_named_result(
        self,
    ) -> None:
        completion = bindings.ActivationCompletion(threading.Event())
        handle = bindings._build_completion_handler(completion)
        self.addCleanup(handle.close)

        def complete(
            operation, index, result_type, argument_types, status, interface
        ):
            status._obj.value = 0
            interface._obj.value = 123
            return 0

        with patch.object(bindings, '_wincall', side_effect=complete):
            self.assertEqual(handle.callbacks[-1](None, None), 0)
        self.assertTrue(completion.event.is_set())
        self.assertEqual(completion.result.status, 0)
        self.assertEqual(completion.result.interface.value, 123)
        self.assertTrue(handle.handler.lpVtbl)

    def test_callback_failure_signals_completion_and_records_a_diagnostic(
        self,
    ) -> None:
        completion = bindings.ActivationCompletion(threading.Event())
        handle = bindings._build_completion_handler(completion)
        self.addCleanup(handle.close)
        with (
            patch.object(
                bindings, '_wincall', side_effect=OSError('device failed')
            ),
            self.assertLogs(bindings.__name__, level='ERROR'),
        ):
            self.assertEqual(handle.callbacks[-1](None, None), 0)
        self.assertTrue(completion.event.is_set())
        self.assertEqual(completion.result.status, 0x80004005)
        self.assertIsNone(completion.result.interface)
