from __future__ import annotations

import ctypes
import logging
import uuid
from collections.abc import Callable
from functools import partial
from typing import Any

from infrastructure.process_session_catalog_parts.windows_types import (
    GUID,
    PROPERTYKEY,
    PROPVARIANT,
)

_logger = logging.getLogger(__name__)


def _hresult_type() -> Any:
    import ctypes

    return getattr(ctypes, 'HRESULT', ctypes.c_long)


def make_guid(raw: str) -> GUID:
    data = uuid.UUID(raw).bytes_le
    guid = GUID()
    guid.Data1 = int.from_bytes(data[0:4], 'little')
    guid.Data2 = int.from_bytes(data[4:6], 'little')
    guid.Data3 = int.from_bytes(data[6:8], 'little')
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

CLSID_MM_DEVICE_ENUMERATOR = make_guid('{BCDE0395-E52F-467C-8E3D-C4579291692E}')
IID_IMM_DEVICE_ENUMERATOR = make_guid('{A95664D2-9614-4F35-A746-DE8DB63617E6}')
IID_IAUDIO_SESSION_MANAGER2 = make_guid(
    '{77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F}'
)
IID_IAUDIO_SESSION_CONTROL2 = make_guid(
    '{BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D}'
)
PKEY_DEVICE_FRIENDLY_NAME = property_key(
    '{A45C254E-DF1C-4EFD-8020-67D146A850E0}', 14
)
HRESULT = getattr(ctypes, 'HRESULT', ctypes.c_long)


def failed(hr: int) -> bool:
    return int(hr) < 0


def device_id(device: ctypes.c_void_p, *, ole32: Any) -> str:
    raw = ctypes.c_void_p()
    hr = _wincall(
        device,
        5,
        HRESULT,
        [ctypes.POINTER(ctypes.c_void_p)],
        ctypes.byref(raw),
    )
    if failed(hr) or not raw.value:
        return ''
    try:
        return str(ctypes.wstring_at(raw.value))
    finally:
        ole32.CoTaskMemFree(raw)


def device_label(device: ctypes.c_void_p, fallback: str, *, ole32: Any) -> str:
    store = ctypes.c_void_p()
    hr = _wincall(
        device,
        4,
        HRESULT,
        [ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)],
        STGM_READ,
        ctypes.byref(store),
    )
    if failed(hr) or not store:
        return fallback
    value = PROPVARIANT()
    try:
        hr = _wincall(
            store,
            5,
            HRESULT,
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
            _logger.debug(
                'Best-effort operation failed in device_label',
                exc_info=True,
            )
        _com_release(store)


def sessions_for_device(
    device: ctypes.c_void_p,
    endpoint_id: str,
    endpoint_label: str,
    *,
    ole32: Any,
) -> list[dict[str, Any]]:
    manager = ctypes.c_void_p()
    hr = _wincall(
        device,
        3,
        HRESULT,
        [
            ctypes.POINTER(GUID),
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ],
        ctypes.byref(IID_IAUDIO_SESSION_MANAGER2),
        CLSCTX_ALL,
        None,
        ctypes.byref(manager),
    )
    if failed(hr) or not manager:
        return []

    session_enum = ctypes.c_void_p()
    try:
        hr = _wincall(
            manager,
            5,
            HRESULT,
            [ctypes.POINTER(ctypes.c_void_p)],
            ctypes.byref(session_enum),
        )
        if failed(hr) or not session_enum:
            return []

        count = ctypes.c_int(0)
        hr = _wincall(
            session_enum,
            3,
            HRESULT,
            [ctypes.POINTER(ctypes.c_int)],
            ctypes.byref(count),
        )
        if failed(hr):
            return []

        by_pid: dict[int, dict[str, Any]] = {}
        for index in range(max(0, int(count.value))):
            control = ctypes.c_void_p()
            hr = _wincall(
                session_enum,
                4,
                HRESULT,
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
                    HRESULT,
                    [
                        ctypes.POINTER(GUID),
                        ctypes.POINTER(ctypes.c_void_p),
                    ],
                    ctypes.byref(IID_IAUDIO_SESSION_CONTROL2),
                    ctypes.byref(control2),
                )
            finally:
                _com_release(control)
            if failed(hr) or not control2:
                continue

            pid = ctypes.c_ulong(0)
            try:
                hr = _wincall(
                    control2,
                    14,
                    HRESULT,
                    [ctypes.POINTER(ctypes.c_ulong)],
                    ctypes.byref(pid),
                )
            finally:
                _com_release(control2)
            if failed(hr) or pid.value == 0:
                continue

            pid_value = int(pid.value)
            if pid_value not in by_pid:
                by_pid[pid_value] = {
                    'pid': pid_value,
                    'label': _process_label(pid_value),
                    'streams': 0,
                    'endpointId': endpoint_id,
                    'endpointLabel': endpoint_label,
                }
            by_pid[pid_value]['streams'] += 1

        return sorted(
            by_pid.values(),
            key=lambda item: str(item.get('label', '')).lower(),
        )
    finally:
        if session_enum:
            _com_release(session_enum)
        _com_release(manager)


def list_windows_session_groups() -> list[dict[str, Any]]:
    try:
        ole32 = ctypes.windll.ole32
        ole32.CoInitializeEx.restype = HRESULT
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ole32.CoCreateInstance.restype = HRESULT
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
        ole32.PropVariantClear.restype = HRESULT

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
                HRESULT,
                [
                    ctypes.c_int,
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_void_p),
                ],
                E_RENDER,
                DEVICE_STATE_ACTIVE,
                ctypes.byref(collection),
            )
            if failed(hr) or not collection:
                return []

            return _enumerate_windows_device_groups(
                collection,
                partial(sessions_for_device, ole32=ole32),
                partial(device_id, ole32=ole32),
                partial(device_label, ole32=ole32),
            )
        finally:
            if collection:
                _com_release(collection)
            if device_enum:
                _com_release(device_enum)
            if initialized:
                try:
                    ole32.CoUninitialize()
                except Exception:
                    _logger.debug(
                        'Best-effort operation failed in list_windows_session_groups',
                        exc_info=True,
                    )
    except Exception:
        _logger.debug(
            'Unable to enumerate Windows audio sessions', exc_info=True
        )
        return []


def _process_label(pid: int) -> str:
    try:
        import psutil

        return psutil.Process(int(pid)).name().removesuffix('.exe')
    except Exception:
        return f'PID {int(pid)}'


def _enumerate_windows_device_groups(
    collection: Any,
    sessions_fn: Callable[[Any, str, str], list[dict[str, Any]]],
    device_id_fn: Callable[[Any], str],
    device_label_fn: Callable[[Any, str], str],
) -> list:
    import ctypes

    HRESULT = _hresult_type()
    count = ctypes.c_uint(0)
    hr = _wincall(
        collection,
        3,
        HRESULT,
        [ctypes.POINTER(ctypes.c_uint)],
        ctypes.byref(count),
    )
    if int(hr) < 0:
        return []

    groups = []
    for index in range(int(count.value)):
        device = ctypes.c_void_p()
        hr = _wincall(
            collection,
            4,
            HRESULT,
            [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
            index,
            ctypes.byref(device),
        )
        if int(hr) < 0 or not device:
            continue
        try:
            fallback = f'Output {index + 1}'
            endpoint_label = device_label_fn(device, fallback)
            endpoint_id = device_id_fn(device) or f'endpoint:{index}'
            sessions = sessions_fn(device, endpoint_id, endpoint_label)
            if sessions:
                groups.append(
                    {
                        'id': endpoint_id,
                        'label': endpoint_label,
                        'sessions': sessions,
                    }
                )
        finally:
            _com_release(device)
    return sorted(groups, key=lambda group: str(group.get('label', '')).lower())


def _wincall(
    obj: Any, idx: int, restype: Any, argtypes: list, *args: Any
) -> Any:
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
        _logger.debug(
            'Best-effort operation failed in _com_release', exc_info=True
        )
