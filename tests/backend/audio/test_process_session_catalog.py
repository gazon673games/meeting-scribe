from __future__ import annotations

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from infrastructure import process_session_catalog
from infrastructure.process_session_catalog_parts import linux, windows


class ProcessSessionCatalogTests(unittest.TestCase):
    def test_linux_sessions_group_streams_by_pid_and_skip_unlabeled_entries(self) -> None:
        pactl_output = """
Sink Input #10
    application.name = "Browser"
    application.process.id = "101"
Sink Input #11
    application.name = "Browser"
    application.process.id = "101"
Sink Input #12
    application.process.id = "202"
Sink Input #13
    application.name = "Player"
"""
        result = SimpleNamespace(returncode=0, stdout=pactl_output)

        with patch("subprocess.run", return_value=result):
            sessions = linux.list_linux_sessions()

        self.assertEqual(sessions[0], {"pid": 101, "label": "Browser", "streams": 2, "index": 10})
        self.assertEqual(sessions[1], {"pid": 0, "label": "Player", "streams": 1, "index": 13})

    def test_process_session_catalog_switches_by_platform(self) -> None:
        with (
            patch.object(process_session_catalog.sys, "platform", "linux"),
            patch("infrastructure.process_session_catalog.list_linux_sessions", return_value=[{"pid": 1}]),
        ):
            self.assertTrue(process_session_catalog.is_per_process_audio_supported() in (True, False))
            self.assertEqual(
                process_session_catalog.list_process_session_groups(),
                [{"id": "linux-default", "label": "Default output", "sessions": [{"pid": 1}]}],
            )
            self.assertEqual(process_session_catalog.list_process_sessions(), [{"pid": 1}])

        with (
            patch.object(process_session_catalog.sys, "platform", "win32"),
            patch.object(process_session_catalog.sys, "getwindowsversion", return_value=SimpleNamespace(major=10, build=20348), create=True),
            patch("infrastructure.process_session_catalog.list_windows_session_groups", return_value=[{"sessions": [{"pid": 2}]}]),
        ):
            self.assertTrue(process_session_catalog.is_per_process_audio_supported())
            self.assertEqual(process_session_catalog.list_process_sessions(), [{"pid": 2}])

        with patch.object(process_session_catalog.sys, "platform", "darwin"):
            self.assertFalse(process_session_catalog.is_per_process_audio_supported())
            self.assertEqual(process_session_catalog.list_process_session_groups(), [])

    def test_windows_session_helpers_fallback_process_labels_and_sort_groups(self) -> None:
        collection = object()
        devices = [object(), object()]

        def fake_wincall(obj, idx, restype, argtypes, *args):  # noqa: ANN001
            if idx == 3:
                args[0]._obj.value = 2
                return 0
            if idx == 4:
                args[1]._obj.value = id(devices[int(args[0])])
                return 0
            if idx == 2:
                return 0
            return -1

        had_hresult = hasattr(ctypes, "HRESULT")
        original_hresult = getattr(ctypes, "HRESULT", None)
        if had_hresult:
            delattr(ctypes, "HRESULT")
        try:
            with patch("infrastructure.process_session_catalog_parts.windows._wincall", side_effect=fake_wincall):
                groups = windows._enumerate_windows_device_groups(
                    collection,
                    lambda device, endpoint_id, endpoint_label: [{"pid": 10}] if endpoint_label == "Beta" else [],
                    lambda device: "endpoint-beta" if device else "",
                    lambda device, fallback: "Beta" if fallback == "Output 1" else "Alpha",
                )
        finally:
            if had_hresult:
                ctypes.HRESULT = original_hresult

        self.assertEqual(groups, [{"id": "endpoint-beta", "label": "Beta", "sessions": [{"pid": 10}]}])
        with patch.dict("sys.modules", {"psutil": None}):
            self.assertEqual(windows._process_label(42), "PID 42")


if __name__ == "__main__":
    unittest.main()
