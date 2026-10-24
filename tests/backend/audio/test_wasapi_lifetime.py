from __future__ import annotations

import ctypes
import threading
import unittest
from contextlib import ExitStack
from unittest.mock import Mock, patch

from audio.infrastructure.sources import wasapi_bindings as bindings


class WasapiLifetimeTests(unittest.TestCase):
    def test_late_callback_retains_handler_and_releases_abandoned_interface(
        self,
    ):
        completion = bindings.ActivationCompletion(threading.Event())
        handle = bindings._build_completion_handler(completion)
        self.addCleanup(handle.close)
        address = ctypes.addressof(handle.handler)
        self.assertEqual(handle.callbacks[1](address), 2)
        completion.abandoned = True
        handle.close()
        self.assertIn(address, bindings._COMPLETIONS)

        def complete(operation, index, result_type, types, status, interface):
            status._obj.value = 0
            interface._obj.value = 123
            return 0

        with (
            patch.object(bindings, '_wincall', side_effect=complete),
            patch.object(bindings, '_com_release') as release_interface,
        ):
            handle.callbacks[-1](address, None)
            self.assertEqual(release_interface.call_args.args[0].value, 123)
        self.assertIsNone(completion.result)
        self.assertTrue(completion.event.is_set())
        self.assertEqual(handle.callbacks[2](address), 0)
        self.assertNotIn(address, bindings._COMPLETIONS)

    def test_query_interface_rejects_unknown_iid_and_counts_owned_reference(
        self,
    ):
        completion = bindings.ActivationCompletion(threading.Event())
        handle = bindings._build_completion_handler(completion)
        self.addCleanup(handle.close)
        address = ctypes.addressof(handle.handler)
        pointer = ctypes.c_void_p(123)
        unknown = bindings._make_guid('{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE}')
        query = handle.callbacks[0]
        status = query(address, ctypes.byref(unknown), ctypes.byref(pointer))
        self.assertEqual(status & 0xFFFFFFFF, 0x80004002)
        self.assertIsNone(pointer.value)
        status = query(
            address, ctypes.byref(bindings.IID_IUnknown), ctypes.byref(pointer)
        )
        self.assertEqual(status, 0)
        self.assertEqual(pointer.value, address)
        self.assertEqual(handle.references, 2)
        self.assertEqual(handle.callbacks[2](address), 1)

    def test_failed_client_start_releases_acquired_capture_interface(self):
        released = []
        capture = Mock()

        def call(client, index, result_type, types, *args):
            if index == 14:
                args[1]._obj.value = 222
            return -1 if index == 10 else 0

        with (
            patch.object(bindings, '_wincall', side_effect=call),
            patch.object(
                bindings,
                '_com_release',
                side_effect=lambda p: released.append(p.value),
            ),
            self.assertRaisesRegex(OSError, 'Start failed'),
            ExitStack() as cleanup,
        ):
            client = ctypes.c_void_p(111)
            cleanup.callback(bindings._com_release, client)
            bindings._capture_client(client, capture, cleanup)
        self.assertEqual(released, [222, 111])
        capture.assert_not_called()

    def test_failed_com_initialization_does_not_uninitialize_foreign_apartment(
        self,
    ):
        ole32 = Mock()
        ole32.CoInitializeEx.return_value = -2147417850
        with (
            patch.object(ctypes, 'windll', Mock(ole32=ole32)),
            self.assertRaisesRegex(OSError, 'CoInitializeEx failed'),
        ):
            bindings.capture_process_loopback(123, Mock())
        ole32.CoUninitialize.assert_not_called()
