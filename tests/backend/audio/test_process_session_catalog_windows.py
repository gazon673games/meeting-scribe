from __future__ import annotations

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from infrastructure.process_session_catalog_parts import windows


class _FakeComFunction:
    def __init__(self, impl) -> None:  # noqa: ANN001
        self.impl = impl
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):  # noqa: ANN002
        return self.impl(*args)


def _set_pointer(arg, value: int) -> None:  # noqa: ANN001
    arg._obj.value = int(value)  # noqa: SLF001


class WindowsProcessSessionCatalogTests(unittest.TestCase):
    def test_wincall_invokes_vtable_slot_with_object_pointer(self) -> None:
        calls: list[tuple[object, object, tuple[object, ...]]] = []

        def fake_cast(obj, pointer_type):  # noqa: ANN001, ARG001
            if obj == "obj":
                return ["vtbl"]
            return {7: "fn"}

        def fake_winfunctype(restype, first_arg, *argtypes):  # noqa: ANN001
            self.assertIs(restype, int)
            self.assertIs(first_arg, ctypes.c_void_p)
            self.assertEqual(argtypes, (str,))

            def bind(fn_ptr):  # noqa: ANN001
                def call(obj, *args):  # noqa: ANN001, ANN002
                    calls.append((fn_ptr, obj, args))
                    return 123

                return call

            return bind

        with (
            patch("ctypes.cast", side_effect=fake_cast),
            patch.object(ctypes, "WINFUNCTYPE", fake_winfunctype, create=True),
        ):
            self.assertEqual(windows._wincall("obj", 7, int, [str], "arg"), 123)  # noqa: SLF001

        self.assertEqual(calls, [("fn", "obj", ("arg",))])

    def test_list_windows_session_groups_walks_com_devices_and_groups_streams(self) -> None:
        def co_create_instance(*args):  # noqa: ANN001
            _set_pointer(args[-1], 111)
            return 0

        fake_ole32 = SimpleNamespace(
            CoInitializeEx=_FakeComFunction(lambda *args: 0),
            CoCreateInstance=_FakeComFunction(co_create_instance),
            CoTaskMemFree=_FakeComFunction(lambda *args: None),
            CoUninitialize=lambda: None,
            PropVariantClear=_FakeComFunction(lambda *args: 0),
        )

        def fake_wincall(obj, idx, restype, argtypes, *args):  # noqa: ANN001, ARG001
            value = int(getattr(obj, "value", obj) or 0)
            if idx == 2:
                return 0
            if value == 111 and idx == 3:
                _set_pointer(args[2], 222)
                return 0
            if value == 222 and idx == 3:
                args[0]._obj.value = 2  # noqa: SLF001
                return 0
            if value == 222 and idx == 4:
                _set_pointer(args[1], 300 + int(getattr(args[0], "value", args[0])))
                return 0
            if value in {300, 301} and idx == 4:
                _set_pointer(args[1], 800 + (value - 300))
                return 0
            if value in {300, 301} and idx == 5:
                return -1
            if value in {800, 801} and idx == 5:
                args[1]._obj.vt = 31  # noqa: SLF001
                args[1]._obj.pwszVal = f"Output Label {value - 799}"  # noqa: SLF001
                return 0
            if value in {300, 301} and idx == 3:
                _set_pointer(args[3], 400 + (value - 300))
                return 0
            if value in {400, 401} and idx == 5:
                _set_pointer(args[0], 500 + (value - 400))
                return 0
            if value in {500, 501} and idx == 3:
                args[0]._obj.value = 2 if value == 500 else 1  # noqa: SLF001
                return 0
            if value in {500, 501} and idx == 4:
                _set_pointer(args[1], (600 if value == 500 else 610) + int(getattr(args[0], "value", args[0])))
                return 0
            if value in {600, 601, 610} and idx == 0:
                _set_pointer(args[1], value + 100)
                return 0
            if value in {700, 701, 710} and idx == 14:
                pid_by_control = {700: 1234, 701: 1234, 710: 456}
                args[0]._obj.value = pid_by_control[value]  # noqa: SLF001
                return 0
            return -1

        with (
            patch.object(ctypes, "windll", SimpleNamespace(ole32=fake_ole32), create=True),
            patch("infrastructure.process_session_catalog_parts.windows._wincall", side_effect=fake_wincall),
            patch("infrastructure.process_session_catalog_parts.windows._process_label", side_effect=lambda pid: f"App {pid}"),
        ):
            groups = windows.list_windows_session_groups()

        self.assertEqual(
            groups,
            [
                {
                    "id": "endpoint:0",
                    "label": "Output Label 1",
                    "sessions": [{"pid": 1234, "label": "App 1234", "streams": 2, "endpointId": "endpoint:0", "endpointLabel": "Output Label 1"}],
                },
                {
                    "id": "endpoint:1",
                    "label": "Output Label 2",
                    "sessions": [{"pid": 456, "label": "App 456", "streams": 1, "endpointId": "endpoint:1", "endpointLabel": "Output Label 2"}],
                },
            ],
        )

    def test_process_label_uses_psutil_name_without_exe_suffix(self) -> None:
        class FakeProcess:
            def __init__(self, pid: int) -> None:
                self.pid = pid

            def name(self) -> str:
                return "Browser.exe"

        fake_psutil = SimpleNamespace(Process=FakeProcess)
        with patch.dict("sys.modules", {"psutil": fake_psutil}):
            self.assertEqual(windows._process_label(99), "Browser")  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
