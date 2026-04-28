from __future__ import annotations

import ctypes
import ctypes.wintypes
import threading
import time
import uuid
from typing import Any, Callable, Optional

import numpy as np

from audio.domain.formats import AudioFormat
from audio.infrastructure.sources.base import BaseSource


# ─── COM / WASAPI constants ───────────────────────────────────────────────────

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

VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_uint8 * 8),
    ]


def _make_guid(s: str) -> GUID:
    b = uuid.UUID(s).bytes_le
    g = GUID()
    g.Data1 = int.from_bytes(b[0:4], "little")
    g.Data2 = int.from_bytes(b[4:6], "little")
    g.Data3 = int.from_bytes(b[6:8], "little")
    for i in range(8):
        g.Data4[i] = b[8 + i]
    return g


IID_IAudioClient = _make_guid("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")
IID_IAudioCaptureClient = _make_guid("{C8ADBD64-E71E-48A0-A4DE-185C395CD317}")
IID_IActivateAudioInterfaceCompletionHandler = _make_guid("{41D949AB-9862-444A-80F6-C261334DA5EB}")

CLSID_MMDeviceEnumerator = _make_guid("{BCDE0395-E52F-467C-8E3D-C4579291692E}")
IID_IMMDeviceEnumerator = _make_guid("{A95664D2-9614-4F35-A746-DE8DB63617E6}")


# ─── PROPVARIANT (simplified, for VT_BLOB) ───────────────────────────────────

class BLOB(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint32), ("pBlobData", ctypes.c_void_p)]


class PROPVARIANT(ctypes.Union):
    class _blob_inner(ctypes.Structure):
        _fields_ = [("vt", ctypes.c_ushort), ("pad", ctypes.c_ushort * 3), ("blob", BLOB)]
    _anonymous_ = ("_inner",)
    _fields_ = [("_inner", _blob_inner), ("_raw", ctypes.c_byte * 24)]


# ─── AUDIOCLIENT_ACTIVATION_PARAMS ───────────────────────────────────────────

class AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(ctypes.Structure):
    _fields_ = [
        ("TargetProcessId", ctypes.c_uint32),
        ("ProcessLoopbackMode", ctypes.c_uint32),
    ]


class AUDIOCLIENT_ACTIVATION_PARAMS(ctypes.Structure):
    _fields_ = [
        ("ActivationType", ctypes.c_uint32),
        ("ProcessLoopbackParams", AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS),
    ]


# ─── WAVEFORMATEX ─────────────────────────────────────────────────────────────

class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ("wFormatTag", ctypes.c_ushort),
        ("nChannels", ctypes.c_ushort),
        ("nSamplesPerSec", ctypes.c_ulong),
        ("nAvgBytesPerSec", ctypes.c_ulong),
        ("nBlockAlign", ctypes.c_ushort),
        ("wBitsPerSample", ctypes.c_ushort),
        ("cbSize", ctypes.c_ushort),
    ]


WAVE_FORMAT_IEEE_FLOAT = 3
WAVE_FORMAT_PCM = 1


# ─── COM vtable helpers ───────────────────────────────────────────────────────

def _wincall(obj: Any, idx: int, restype: Any, argtypes: list, *args: Any) -> int:
    vtbl = ctypes.cast(
        ctypes.cast(obj, ctypes.POINTER(ctypes.c_void_p))[0],
        ctypes.POINTER(ctypes.c_void_p),
    )
    actual_restype = ctypes.c_long if restype is ctypes.HRESULT else restype
    proto = ctypes.WINFUNCTYPE(actual_restype, ctypes.c_void_p, *argtypes)
    return proto(vtbl[idx])(obj, *args)


def _com_release(obj: Any) -> None:
    try:
        _wincall(obj, 2, ctypes.c_ulong, [])
    except Exception:
        pass


# ─── IActivateAudioInterfaceCompletionHandler COM impl ───────────────────────

class _CompletionVtbl(ctypes.Structure):
    _fields_ = [
        ("QueryInterface", ctypes.c_void_p),
        ("AddRef", ctypes.c_void_p),
        ("Release", ctypes.c_void_p),
        ("ActivateCompleted", ctypes.c_void_p),
    ]


class _CompletionHandler(ctypes.Structure):
    _fields_ = [("lpVtbl", ctypes.POINTER(_CompletionVtbl))]


def _build_completion_handler(event: threading.Event, result_box: list) -> tuple:
    """Return (handler_struct, vtbl, keep_alive_callbacks) for COM completion handler."""

    HRESULT = ctypes.HRESULT
    QI_TYPE = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
    ULONG_TYPE = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
    COMPLETE_TYPE = ctypes.WINFUNCTYPE(HRESULT, ctypes.c_void_p, ctypes.c_void_p)

    @QI_TYPE
    def qi(this, riid, ppv):
        ppv[0] = this
        return 0

    @ULONG_TYPE
    def add_ref(this):
        return 2

    @ULONG_TYPE
    def release(this):
        return 1

    @COMPLETE_TYPE
    def activate_completed(this, op):
        try:
            hr_result = ctypes.HRESULT(0)
            p_iface = ctypes.c_void_p()
            _wincall(op, 3, ctypes.HRESULT,
                     [ctypes.POINTER(ctypes.HRESULT), ctypes.POINTER(ctypes.c_void_p)],
                     ctypes.byref(hr_result), ctypes.byref(p_iface))
            result_box.append((int(hr_result.value), p_iface))
        except Exception as exc:
            result_box.append((0x80004005, None))
        finally:
            event.set()
        return 0

    vtbl = _CompletionVtbl()
    vtbl.QueryInterface = ctypes.cast(qi, ctypes.c_void_p)
    vtbl.AddRef = ctypes.cast(add_ref, ctypes.c_void_p)
    vtbl.Release = ctypes.cast(release, ctypes.c_void_p)
    vtbl.ActivateCompleted = ctypes.cast(activate_completed, ctypes.c_void_p)

    handler = _CompletionHandler()
    handler.lpVtbl = ctypes.pointer(vtbl)

    return handler, vtbl, (qi, add_ref, release, activate_completed)


# ─── ProcessLoopbackWinSource ─────────────────────────────────────────────────

class ProcessLoopbackWinSource(BaseSource):
    """Per-process audio capture using WASAPI Process Loopback (Windows 10 2004+)."""

    def __init__(self, name: str, pid: int):
        fmt = AudioFormat(sample_rate=44100, channels=2, dtype="float32", blocksize=1024)
        super().__init__(name=name, format=fmt)
        self.pid = int(pid)
        self._on_audio: Optional[Callable[[str, np.ndarray], None]] = None
        self._on_error: Optional[Callable[[str, str], None]] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._err_lock = threading.Lock()
        self._last_error: Optional[str] = None

    def set_error_callback(self, cb: Optional[Callable[[str, str], None]]) -> None:
        self._on_error = cb

    def get_last_error(self) -> Optional[str]:
        with self._err_lock:
            return self._last_error

    def _set_last_error(self, text: Optional[str]) -> None:
        with self._err_lock:
            self._last_error = text

    def start(self, on_audio: Callable[[str, np.ndarray], None]) -> None:
        self._on_audio = on_audio
        self._set_last_error(None)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-procloop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run(self) -> None:
        try:
            ole32 = ctypes.windll.ole32
            ole32.CoInitializeEx(None, COINIT_MULTITHREADED)

            # Build activation params
            ap = AUDIOCLIENT_ACTIVATION_PARAMS()
            ap.ActivationType = AUDCLNT_ACTIVATION_TYPE_PROCESS_LOOPBACK
            ap.ProcessLoopbackParams.TargetProcessId = self.pid
            ap.ProcessLoopbackParams.ProcessLoopbackMode = PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE

            pv = PROPVARIANT()
            pv.vt = VT_BLOB
            pv.blob.cbSize = ctypes.sizeof(AUDIOCLIENT_ACTIVATION_PARAMS)
            pv.blob.pBlobData = ctypes.cast(ctypes.pointer(ap), ctypes.c_void_p)

            completion_event = threading.Event()
            result_box: list = []
            handler, vtbl, callbacks = _build_completion_handler(completion_event, result_box)

            try:
                mmdevapi = ctypes.WinDLL("Mmdevapi.dll")
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
                    ctypes.byref(handler),
                    ctypes.byref(p_async_op),
                )
                if hr != S_OK:
                    raise OSError(f"ActivateAudioInterfaceAsync failed: 0x{hr & 0xFFFFFFFF:08x}")
            except Exception as exc:
                raise

            if not completion_event.wait(timeout=5.0) or not result_box:
                raise TimeoutError("Audio interface activation timed out")

            hr_activate, p_client = result_box[0]
            if hr_activate != S_OK or not p_client:
                raise OSError(f"Activation result: 0x{hr_activate & 0xFFFFFFFF:08x}")

            # Build WAVEFORMATEX for 48kHz float32 stereo
            wfx = WAVEFORMATEX()
            wfx.wFormatTag = WAVE_FORMAT_PCM
            wfx.nChannels = 2
            wfx.nSamplesPerSec = 44100
            wfx.wBitsPerSample = 16
            wfx.nBlockAlign = wfx.nChannels * (wfx.wBitsPerSample // 8)
            wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign
            wfx.cbSize = 0

            # IAudioClient::Initialize(SHARED, 0, 200ms, 0, &wfx, NULL) → vtable[3]
            stream_flags = AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM
            hr = _wincall(p_client, 3, ctypes.HRESULT,
                          [ctypes.c_int, ctypes.c_uint32, ctypes.c_longlong,
                           ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p],
                          AUDCLNT_SHAREMODE_SHARED, stream_flags, 0, 0,
                          ctypes.byref(wfx), None)
            if hr != S_OK:
                raise OSError(f"IAudioClient::Initialize failed: 0x{hr & 0xFFFFFFFF:08x}")

            # GetService(IID_IAudioCaptureClient) → vtable[14]
            p_capture = ctypes.c_void_p()
            hr = _wincall(p_client, 14, ctypes.HRESULT,
                          [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)],
                          ctypes.byref(IID_IAudioCaptureClient), ctypes.byref(p_capture))
            if hr != S_OK:
                raise OSError(f"GetService(IAudioCaptureClient) failed")

            # Start → vtable[9]
            _wincall(p_client, 10, ctypes.HRESULT, [])

            try:
                self._capture_loop(p_capture, wfx)
            finally:
                _wincall(p_client, 11, ctypes.HRESULT, [])  # Stop
                _com_release(p_capture)
                _com_release(p_client)

        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            self._set_last_error(err)
            if self._on_error:
                try:
                    self._on_error(self.name, err)
                except Exception:
                    pass

    def _capture_loop(self, p_capture: Any, wfx: WAVEFORMATEX) -> None:
        frame_bytes = wfx.nBlockAlign

        while not self._stop.is_set():
            packet_frames = ctypes.c_uint32(0)
            hr = _wincall(p_capture, 5, ctypes.HRESULT, [ctypes.POINTER(ctypes.c_uint32)], ctypes.byref(packet_frames))
            if hr != S_OK or packet_frames.value == 0:
                time.sleep(0.005)
                continue
            # IAudioCaptureClient::GetBuffer(&pData, &numFrames, &flags, NULL, NULL) → vtable[3]
            p_data = ctypes.c_void_p()
            num_frames = ctypes.c_uint32(0)
            flags = ctypes.c_uint32(0)
            hr = _wincall(p_capture, 3, ctypes.HRESULT,
                          [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint32),
                           ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p, ctypes.c_void_p],
                          ctypes.byref(p_data), ctypes.byref(num_frames),
                          ctypes.byref(flags), None, None)

            if hr != S_OK or num_frames.value == 0:
                time.sleep(0.005)
                continue

            n = num_frames.value
            raw = (ctypes.c_byte * (n * frame_bytes)).from_address(p_data.value)
            if wfx.wFormatTag == WAVE_FORMAT_PCM and wfx.wBitsPerSample == 16:
                arr = np.frombuffer(bytes(raw), dtype=np.int16).reshape(n, wfx.nChannels).astype(np.float32) / 32768.0
            else:
                arr = np.frombuffer(bytes(raw), dtype=np.float32).reshape(n, wfx.nChannels)

            # ReleaseBuffer → vtable[4]
            _wincall(p_capture, 4, ctypes.HRESULT, [ctypes.c_uint32], n)

            if self._on_audio is not None and arr.shape[0] > 0:
                self._on_audio(self.name, arr.astype(np.float32))
