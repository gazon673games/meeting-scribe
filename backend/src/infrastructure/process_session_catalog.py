from __future__ import annotations

import sys
from typing import Any, Dict, List


def is_per_process_audio_supported() -> bool:
    if sys.platform == "win32":
        try:
            version = sys.getwindowsversion()
            return version.major > 10 or (version.major == 10 and version.build >= 20348)
        except Exception:
            return False
    if sys.platform == "linux":
        import shutil

        return bool(shutil.which("pactl") and shutil.which("parec"))
    return False


def list_process_sessions() -> List[Dict[str, Any]]:
    return [session for group in list_process_session_groups() for session in group.get("sessions", [])]


def list_process_session_groups() -> List[Dict[str, Any]]:
    if sys.platform == "win32":
        return _list_windows_session_groups()
    if sys.platform == "linux":
        sessions = _list_linux_sessions()
        return [{"id": "linux-default", "label": "Default output", "sessions": sessions}] if sessions else []
    return []


def _list_windows_session_groups() -> List[Dict[str, Any]]:
    try:
        import ctypes
        import uuid

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", ctypes.c_uint32),
                ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_uint8 * 8),
            ]

        class PROPERTYKEY(ctypes.Structure):
            _fields_ = [("fmtid", GUID), ("pid", ctypes.c_uint32)]

        class PROPVARIANT_VALUE(ctypes.Union):
            _fields_ = [("pwszVal", ctypes.c_wchar_p), ("raw", ctypes.c_byte * 16)]

        class PROPVARIANT(ctypes.Structure):
            _anonymous_ = ("value",)
            _fields_ = [
                ("vt", ctypes.c_ushort),
                ("wReserved1", ctypes.c_ushort),
                ("wReserved2", ctypes.c_ushort),
                ("wReserved3", ctypes.c_ushort),
                ("value", PROPVARIANT_VALUE),
            ]

        def make_guid(raw: str) -> GUID:
            data = uuid.UUID(raw).bytes_le
            guid = GUID()
            guid.Data1 = int.from_bytes(data[0:4], "little")
            guid.Data2 = int.from_bytes(data[4:6], "little")
            guid.Data3 = int.from_bytes(data[6:8], "little")
            for pos in range(8):
                guid.Data4[pos] = data[8 + pos]
            return guid

        def property_key(raw: str, pid: int) -> PROPERTYKEY:
            key = PROPERTYKEY()
            key.fmtid = make_guid(raw)
            key.pid = int(pid)
            return key

        CLSCTX_ALL = 23
        DEVICE_STATE_ACTIVE = 1
        STGM_READ = 0
        VT_LPWSTR = 31
        E_RENDER = 0

        CLSID_MM_DEVICE_ENUMERATOR = make_guid("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
        IID_IMM_DEVICE_ENUMERATOR = make_guid("{A95664D2-9614-4F35-A746-DE8DB63617E6}")
        IID_IAUDIO_SESSION_MANAGER2 = make_guid("{77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F}")
        IID_IAUDIO_SESSION_CONTROL2 = make_guid("{BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D}")
        PKEY_DEVICE_FRIENDLY_NAME = property_key("{A45C254E-DF1C-4EFD-8020-67D146A850E0}", 14)

        ole32 = ctypes.windll.ole32
        ole32.CoInitializeEx.restype = ctypes.HRESULT
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ole32.CoCreateInstance.restype = ctypes.HRESULT
        ole32.CoCreateInstance.argtypes = [
            ctypes.POINTER(GUID),
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
        ole32.CoTaskMemFree.restype = None
        ole32.PropVariantClear.argtypes = [ctypes.POINTER(PROPVARIANT)]
        ole32.PropVariantClear.restype = ctypes.HRESULT

        def failed(hr: int) -> bool:
            return int(hr) < 0

        def device_id(device: ctypes.c_void_p) -> str:
            raw = ctypes.c_void_p()
            hr = _wincall(device, 5, ctypes.HRESULT, [ctypes.POINTER(ctypes.c_void_p)], ctypes.byref(raw))
            if failed(hr) or not raw.value:
                return ""
            try:
                return str(ctypes.wstring_at(raw.value))
            finally:
                ole32.CoTaskMemFree(raw)

        def device_label(device: ctypes.c_void_p, fallback: str) -> str:
            store = ctypes.c_void_p()
            hr = _wincall(device, 4, ctypes.HRESULT, [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)], STGM_READ, ctypes.byref(store))
            if failed(hr) or not store:
                return fallback
            value = PROPVARIANT()
            try:
                hr = _wincall(
                    store,
                    5,
                    ctypes.HRESULT,
                    [ctypes.POINTER(PROPERTYKEY), ctypes.POINTER(PROPVARIANT)],
                    ctypes.byref(PKEY_DEVICE_FRIENDLY_NAME),
                    ctypes.byref(value),
                )
                if not failed(hr) and value.vt == VT_LPWSTR and value.pwszVal:
                    return str(value.pwszVal)
                return fallback
            finally:
                try:
                    ole32.PropVariantClear(ctypes.byref(value))
                except Exception:
                    pass
                _com_release(store)

        def sessions_for_device(device: ctypes.c_void_p, endpoint_id: str, endpoint_label: str) -> List[Dict[str, Any]]:
            manager = ctypes.c_void_p()
            hr = _wincall(
                device,
                3,
                ctypes.HRESULT,
                [ctypes.POINTER(GUID), ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
                ctypes.byref(IID_IAUDIO_SESSION_MANAGER2),
                CLSCTX_ALL,
                None,
                ctypes.byref(manager),
            )
            if failed(hr) or not manager:
                return []

            session_enum = ctypes.c_void_p()
            try:
                hr = _wincall(manager, 5, ctypes.HRESULT, [ctypes.POINTER(ctypes.c_void_p)], ctypes.byref(session_enum))
                if failed(hr) or not session_enum:
                    return []

                count = ctypes.c_int(0)
                hr = _wincall(session_enum, 3, ctypes.HRESULT, [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
                if failed(hr):
                    return []

                by_pid: Dict[int, Dict[str, Any]] = {}
                for index in range(max(0, int(count.value))):
                    control = ctypes.c_void_p()
                    hr = _wincall(
                        session_enum,
                        4,
                        ctypes.HRESULT,
                        [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)],
                        index,
                        ctypes.byref(control),
                    )
                    if failed(hr) or not control:
                        continue

                    control2 = ctypes.c_void_p()
                    try:
                        hr = _wincall(
                            control,
                            0,
                            ctypes.HRESULT,
                            [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)],
                            ctypes.byref(IID_IAUDIO_SESSION_CONTROL2),
                            ctypes.byref(control2),
                        )
                    finally:
                        _com_release(control)
                    if failed(hr) or not control2:
                        continue

                    pid = ctypes.c_ulong(0)
                    try:
                        hr = _wincall(control2, 14, ctypes.HRESULT, [ctypes.POINTER(ctypes.c_ulong)], ctypes.byref(pid))
                    finally:
                        _com_release(control2)
                    if failed(hr) or pid.value == 0:
                        continue

                    pid_value = int(pid.value)
                    if pid_value not in by_pid:
                        by_pid[pid_value] = {
                            "pid": pid_value,
                            "label": _process_label(pid_value),
                            "streams": 0,
                            "endpointId": endpoint_id,
                            "endpointLabel": endpoint_label,
                        }
                    by_pid[pid_value]["streams"] += 1

                return sorted(by_pid.values(), key=lambda item: str(item.get("label", "")).lower())
            finally:
                if session_enum:
                    _com_release(session_enum)
                _com_release(manager)

        initialized = False
        hr_init = ole32.CoInitializeEx(None, 0)
        if int(hr_init) in (0, 1):
            initialized = True

        device_enum = ctypes.c_void_p()
        collection = ctypes.c_void_p()
        try:
            hr = ole32.CoCreateInstance(
                ctypes.byref(CLSID_MM_DEVICE_ENUMERATOR),
                None,
                CLSCTX_ALL,
                ctypes.byref(IID_IMM_DEVICE_ENUMERATOR),
                ctypes.byref(device_enum),
            )
            if failed(hr) or not device_enum:
                return []

            hr = _wincall(
                device_enum,
                3,
                ctypes.HRESULT,
                [ctypes.c_int, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)],
                E_RENDER,
                DEVICE_STATE_ACTIVE,
                ctypes.byref(collection),
            )
            if failed(hr) or not collection:
                return []

            count = ctypes.c_uint(0)
            hr = _wincall(collection, 3, ctypes.HRESULT, [ctypes.POINTER(ctypes.c_uint)], ctypes.byref(count))
            if failed(hr):
                return []

            groups: List[Dict[str, Any]] = []
            for index in range(int(count.value)):
                device = ctypes.c_void_p()
                hr = _wincall(
                    collection,
                    4,
                    ctypes.HRESULT,
                    [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
                    index,
                    ctypes.byref(device),
                )
                if failed(hr) or not device:
                    continue
                try:
                    fallback = f"Output {index + 1}"
                    endpoint_label = device_label(device, fallback)
                    endpoint_id = device_id(device) or f"endpoint:{index}"
                    sessions = sessions_for_device(device, endpoint_id, endpoint_label)
                    if sessions:
                        groups.append(
                            {
                                "id": endpoint_id,
                                "label": endpoint_label,
                                "sessions": sessions,
                            }
                        )
                finally:
                    _com_release(device)

            return sorted(groups, key=lambda group: str(group.get("label", "")).lower())
        finally:
            if collection:
                _com_release(collection)
            if device_enum:
                _com_release(device_enum)
            if initialized:
                try:
                    ole32.CoUninitialize()
                except Exception:
                    pass
    except Exception:
        return []


def _process_label(pid: int) -> str:
    try:
        import psutil

        return psutil.Process(int(pid)).name().removesuffix(".exe")
    except Exception:
        return f"PID {int(pid)}"


def _wincall(obj: Any, idx: int, restype: Any, argtypes: list, *args: Any) -> Any:
    import ctypes

    vtbl_ptr = ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0]
    vtbl = ctypes.cast(vtbl_ptr, ctypes.POINTER(ctypes.c_void_p))
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    fn = proto(vtbl[idx])
    return fn(obj, *args)


def _com_release(obj: Any) -> None:
    try:
        import ctypes

        _wincall(obj, 2, ctypes.c_ulong, [])
    except Exception:
        pass


def _list_linux_sessions() -> List[Dict[str, Any]]:
    try:
        import re
        import subprocess

        result = subprocess.run(
            ["pactl", "list", "sink-inputs"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        sessions: Dict[str, Dict[str, Any]] = {}
        current: Dict[str, str] = {}
        current_index: str = ""

        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            match = re.match(r"^Sink Input #(\d+)$", line)
            if match:
                if current_index:
                    _merge_linux_session(sessions, current_index, current)
                current_index = match.group(1)
                current = {}
                continue
            match = re.match(r'application\.name\s*=\s*"(.+)"', line)
            if match:
                current["app_name"] = match.group(1)
                continue
            match = re.match(r'application\.process\.id\s*=\s*"(\d+)"', line)
            if match:
                current["pid"] = match.group(1)

        if current_index:
            _merge_linux_session(sessions, current_index, current)

        return [
            {
                "pid": int(value.get("pid", 0)),
                "label": value["app_name"],
                "streams": value["streams"],
                "index": int(value["first_index"]),
            }
            for value in sessions.values()
            if "app_name" in value
        ]
    except Exception:
        return []


def _merge_linux_session(
    sessions: Dict[str, Dict[str, Any]],
    index: str,
    current: Dict[str, str],
) -> None:
    if "app_name" not in current:
        return
    key = current.get("pid") or f"idx_{index}"
    if key not in sessions:
        sessions[key] = {
            "app_name": current["app_name"],
            "pid": current.get("pid", "0"),
            "streams": 0,
            "first_index": index,
        }
    sessions[key]["streams"] += 1
