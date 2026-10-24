from __future__ import annotations

import ctypes
import ctypes.wintypes
import logging
import threading
import uuid
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_STREAMFLAGS_EVENTCALLBACK = 0x00040000
AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM = 0x80000000
AUDCLNT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0
VT_BLOB = 65
COINIT_MULTITHREADED = 0
CLSCTX_ALL = 23
eRender, eConsole = 0, 0
S_OK = 0

VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = 'VAD\\Process_Loopback'


_logger = logging.getLogger(__name__)


class GUID(ctypes.Structure):
    _fields_ = [
        ('Data1', ctypes.c_uint32),
        ('Data2', ctypes.c_uint16),
        ('Data3', ctypes.c_uint16),
        ('Data4', ctypes.c_uint8 * 8),
    ]


def _make_guid(raw: str) -> GUID:
    data = uuid.UUID(raw).bytes_le
    guid = GUID()
    guid.Data1 = int.from_bytes(data[0:4], 'little')
    guid.Data2 = int.from_bytes(data[4:6], 'little')
    guid.Data3 = int.from_bytes(data[6:8], 'little')
    for i in range(8):
        guid.Data4[i] = data[8 + i]
    return guid


IID_IAudioClient = _make_guid('{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}')
IID_IUnknown = _make_guid('{00000000-0000-0000-C000-000000000046}')
IID_IAudioCaptureClient = _make_guid('{C8ADBD64-E71E-48A0-A4DE-185C395CD317}')
IID_IActivateAudioInterfaceCompletionHandler = _make_guid(
    '{41D949AB-9862-444A-80F6-C261334DA5EB}'
)

CLSID_MMDeviceEnumerator = _make_guid('{BCDE0395-E52F-467C-8E3D-C4579291692E}')
IID_IMMDeviceEnumerator = _make_guid('{A95664D2-9614-4F35-A746-DE8DB63617E6}')


class BLOB(ctypes.Structure):
    _fields_ = [('cbSize', ctypes.c_uint32), ('pBlobData', ctypes.c_void_p)]


class PROPVARIANT(ctypes.Union):
    class _blob_inner(ctypes.Structure):
        _fields_ = [
            ('vt', ctypes.c_ushort),
            ('pad', ctypes.c_ushort * 3),
            ('blob', BLOB),
        ]

    _anonymous_ = ('_inner',)
    _fields_ = [('_inner', _blob_inner), ('_raw', ctypes.c_byte * 24)]


class AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(ctypes.Structure):
    _fields_ = [
        ('TargetProcessId', ctypes.c_uint32),
        ('ProcessLoopbackMode', ctypes.c_uint32),
    ]


class AUDIOCLIENT_ACTIVATION_PARAMS(ctypes.Structure):
    _fields_ = [
        ('ActivationType', ctypes.c_uint32),
        ('ProcessLoopbackParams', AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS),
    ]


class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ('wFormatTag', ctypes.c_ushort),
        ('nChannels', ctypes.c_ushort),
        ('nSamplesPerSec', ctypes.c_ulong),
        ('nAvgBytesPerSec', ctypes.c_ulong),
        ('nBlockAlign', ctypes.c_ushort),
        ('wBitsPerSample', ctypes.c_ushort),
        ('cbSize', ctypes.c_ushort),
    ]


WAVE_FORMAT_IEEE_FLOAT = 3
WAVE_FORMAT_PCM = 1


def _wincall(
    obj: Any, idx: int, restype: Any, argtypes: list, *args: Any
) -> int:
    vtbl = ctypes.cast(
        ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0],
        ctypes.POINTER(ctypes.c_void_p),
    )
    actual_restype = ctypes.c_long if restype is ctypes.HRESULT else restype
    proto = ctypes.WINFUNCTYPE(actual_restype, ctypes.c_void_p, *argtypes)
    return proto(vtbl[idx])(obj, *args)


def _com_release(obj: Any) -> None:
    if not obj:
        return
    try:
        _wincall(obj, 2, ctypes.c_ulong, [])
    except Exception:
        _logger.debug(
            'Best-effort operation failed in _com_release', exc_info=True
        )


class _CompletionVtbl(ctypes.Structure):
    _fields_ = [
        ('QueryInterface', ctypes.c_void_p),
        ('AddRef', ctypes.c_void_p),
        ('Release', ctypes.c_void_p),
        ('ActivateCompleted', ctypes.c_void_p),
    ]


class _CompletionHandler(ctypes.Structure):
    _fields_ = [('lpVtbl', ctypes.POINTER(_CompletionVtbl))]


@dataclass(frozen=True)
class ActivationResult:
    status: int
    interface: ctypes.c_void_p | None


@dataclass
class ActivationCompletion:
    event: threading.Event
    result: ActivationResult | None = None
    abandoned: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)


@dataclass
class CompletionHandle:
    handler: _CompletionHandler
    vtable: _CompletionVtbl
    callbacks: tuple[Any, ...]
    activation_parameters: tuple[object, ...] = ()
    marshaler: ctypes.c_void_p | None = None
    references: int = 1
    owner_released: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)

    def close(self) -> None:
        with self.lock:
            if self.owner_released:
                return
            self.owner_released = True
        self.callbacks[2](ctypes.addressof(self.handler))


_COMPLETIONS: dict[int, CompletionHandle] = {}
_COMPLETIONS_LOCK = threading.RLock()


def _build_completion_handler(
    completion: ActivationCompletion,
) -> CompletionHandle:
    """Keep COM callback objects alive for the activation lifetime."""

    # Keep raw HRESULT values; ctypes.HRESULT adds Python-side error checking.
    HRESULT = ctypes.c_long
    QI_TYPE = ctypes.WINFUNCTYPE(
        HRESULT,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    ULONG_TYPE = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
    COMPLETE_TYPE = ctypes.WINFUNCTYPE(
        HRESULT, ctypes.c_void_p, ctypes.c_void_p
    )

    @QI_TYPE
    def qi(this, riid, ppv):
        if not ppv or not riid:
            return 0x80004003  # E_POINTER
        ppv[0] = None
        requested = ctypes.string_at(riid, ctypes.sizeof(GUID))
        if requested in (
            bytes(IID_IUnknown),
            bytes(IID_IActivateAudioInterfaceCompletionHandler),
        ):
            ppv[0] = this
            add_ref(this)
            return S_OK
        if handle.marshaler:
            try:
                return _wincall(
                    handle.marshaler,
                    0,
                    HRESULT,
                    [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
                    riid,
                    ppv,
                )
            except Exception:
                _logger.exception('Cannot query audio completion marshaler')
                return 0x80004005  # E_FAIL
        return 0x80004002  # E_NOINTERFACE

    @ULONG_TYPE
    def add_ref(this):
        with handle.lock:
            handle.references += 1
            return handle.references

    @ULONG_TYPE
    def release(this):
        with handle.lock:
            handle.references -= 1
            references = handle.references
        if references == 0:
            _com_release(handle.marshaler)
            handle.marshaler = None
            with _COMPLETIONS_LOCK:
                _COMPLETIONS.pop(ctypes.addressof(handle.handler), None)
        return references

    @COMPLETE_TYPE
    def activate_completed(this, op):
        p_iface = ctypes.c_void_p()
        try:
            hr_result = ctypes.HRESULT(0)
            status = _wincall(
                op,
                3,
                ctypes.HRESULT,
                [
                    ctypes.POINTER(ctypes.HRESULT),
                    ctypes.POINTER(ctypes.c_void_p),
                ],
                ctypes.byref(hr_result),
                ctypes.byref(p_iface),
            )
            result = ActivationResult(
                int(hr_result.value) if status == S_OK else status, p_iface
            )
            with completion.lock:
                if completion.abandoned:
                    _com_release(p_iface)
                else:
                    completion.result = result
        except Exception:
            _com_release(p_iface)
            _logger.exception('WASAPI activation callback failed')
            with completion.lock:
                completion.result = ActivationResult(0x80004005, None)
        finally:
            completion.event.set()
        return 0

    vtbl = _CompletionVtbl()
    vtbl.QueryInterface = ctypes.cast(qi, ctypes.c_void_p)
    vtbl.AddRef = ctypes.cast(add_ref, ctypes.c_void_p)
    vtbl.Release = ctypes.cast(release, ctypes.c_void_p)
    vtbl.ActivateCompleted = ctypes.cast(activate_completed, ctypes.c_void_p)

    handler = _CompletionHandler()
    handler.lpVtbl = ctypes.pointer(vtbl)

    handle = CompletionHandle(
        handler, vtbl, (qi, add_ref, release, activate_completed)
    )
    with _COMPLETIONS_LOCK:
        _COMPLETIONS[ctypes.addressof(handler)] = handle
    return handle


def capture_process_loopback(
    pid: int,
    capture: Callable[[ctypes.c_void_p, WAVEFORMATEX], None],
) -> None:
    ole32 = ctypes.windll.ole32
    status = ole32.CoInitializeEx(None, COINIT_MULTITHREADED)
    if status not in (S_OK, 1):  # S_FALSE also owns a matching CoUninitialize.
        raise OSError(f'CoInitializeEx failed: 0x{status & 0xFFFFFFFF:08x}')
    with ExitStack() as cleanup:
        cleanup.callback(ole32.CoUninitialize)
        client = _activate_process_client(pid, ole32, cleanup)
        _capture_client(client, capture, cleanup)


def _activate_process_client(
    pid: int, ole32: Any, cleanup: ExitStack
) -> ctypes.c_void_p:

    # Build activation params
    ap = AUDIOCLIENT_ACTIVATION_PARAMS()
    ap.ActivationType = AUDCLNT_ACTIVATION_TYPE_PROCESS_LOOPBACK
    ap.ProcessLoopbackParams.TargetProcessId = pid
    ap.ProcessLoopbackParams.ProcessLoopbackMode = (
        PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE
    )

    pv = PROPVARIANT()
    pv.vt = VT_BLOB
    pv.blob.cbSize = ctypes.sizeof(AUDIOCLIENT_ACTIVATION_PARAMS)
    pv.blob.pBlobData = ctypes.cast(ctypes.pointer(ap), ctypes.c_void_p)

    completion = ActivationCompletion(threading.Event())
    completion_handle = _build_completion_handler(completion)
    completion_handle.activation_parameters = (ap, pv)
    cleanup.callback(completion_handle.close)
    marshaler = ctypes.c_void_p()
    status = ole32.CoCreateFreeThreadedMarshaler(
        ctypes.byref(completion_handle.handler), ctypes.byref(marshaler)
    )
    completion_handle.marshaler = marshaler
    if status != S_OK:
        raise OSError(
            f'Cannot create audio completion marshaler: 0x{status & 0xFFFFFFFF:08x}'
        )

    mmdevapi = ctypes.WinDLL('Mmdevapi.dll')
    fn = mmdevapi.ActivateAudioInterfaceAsync
    fn.restype = ctypes.c_long
    fn.argtypes = [
        ctypes.c_wchar_p,
        ctypes.POINTER(GUID),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    p_async_op = ctypes.c_void_p()
    hr = fn(
        VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
        ctypes.byref(IID_IAudioClient),
        ctypes.byref(pv),
        ctypes.byref(completion_handle.handler),
        ctypes.byref(p_async_op),
    )
    cleanup.callback(_com_release, p_async_op)
    if hr != S_OK:
        raise OSError(
            f'ActivateAudioInterfaceAsync failed: 0x{hr & 0xFFFFFFFF:08x}'
        )

    completion.event.wait(timeout=5.0)
    with completion.lock:
        if completion.result is None:
            # Windows owns a handler reference until its late callback returns.
            completion.abandoned = True
            raise TimeoutError('Audio interface activation timed out')
        hr_activate = completion.result.status
        p_client = completion.result.interface
    cleanup.callback(_com_release, p_client)
    if hr_activate != S_OK or not p_client:
        raise OSError(f'Activation result: 0x{hr_activate & 0xFFFFFFFF:08x}')
    return p_client


def _capture_client(
    p_client: ctypes.c_void_p,
    capture: Callable[[ctypes.c_void_p, WAVEFORMATEX], None],
    cleanup: ExitStack,
) -> None:

    # Capture PCM16 stereo at the declared 44100 Hz device rate.
    wfx = WAVEFORMATEX()
    wfx.wFormatTag = WAVE_FORMAT_PCM
    wfx.nChannels = 2
    wfx.nSamplesPerSec = 44100
    wfx.wBitsPerSample = 16
    wfx.nBlockAlign = wfx.nChannels * (wfx.wBitsPerSample // 8)
    wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign
    wfx.cbSize = 0

    # IAudioClient::Initialize(SHARED, 0, 200ms, 0, &wfx, NULL) -> vtable[3]
    stream_flags = (
        AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
    )
    hr = _wincall(
        p_client,
        3,
        ctypes.HRESULT,
        [
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ],
        AUDCLNT_SHAREMODE_SHARED,
        stream_flags,
        0,
        0,
        ctypes.byref(wfx),
        None,
    )
    if hr != S_OK:
        raise OSError(
            f'IAudioClient::Initialize failed: 0x{hr & 0xFFFFFFFF:08x}'
        )

    # GetService(IID_IAudioCaptureClient) -> vtable[14]
    p_capture = ctypes.c_void_p()
    hr = _wincall(
        p_client,
        14,
        ctypes.HRESULT,
        [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)],
        ctypes.byref(IID_IAudioCaptureClient),
        ctypes.byref(p_capture),
    )
    cleanup.callback(_com_release, p_capture)
    if hr != S_OK:
        raise OSError('GetService(IAudioCaptureClient) failed')

    # IAudioClient::Start
    status = _wincall(p_client, 10, ctypes.HRESULT, [])
    if status != S_OK:
        raise OSError(
            f'IAudioClient::Start failed: 0x{status & 0xFFFFFFFF:08x}'
        )
    cleanup.callback(_stop_client, p_client)
    capture(p_capture, wfx)


def _stop_client(client: ctypes.c_void_p) -> None:
    status = _wincall(client, 11, ctypes.HRESULT, [])
    if status != S_OK:
        _logger.error('IAudioClient::Stop failed: 0x%08x', status & 0xFFFFFFFF)
