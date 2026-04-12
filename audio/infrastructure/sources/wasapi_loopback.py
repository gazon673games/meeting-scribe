from __future__ import annotations

from typing import Any, Callable, Optional
import threading

import numpy as np

# soundcard 0.4.x uses numpy.fromstring() in binary mode, which is removed in NumPy 2.x.
# Provide a local compatibility shim before importing soundcard.
if not getattr(np, "_meeting_scribe_fromstring_compat", False):
    _np_fromstring_orig = np.fromstring

    def _np_fromstring_compat(data, dtype=float, count=-1, sep="", **kwargs):
        if sep == "" and isinstance(data, (bytes, bytearray, memoryview)):
            return np.frombuffer(data, dtype=dtype, count=count)
        try:
            return _np_fromstring_orig(data, dtype=dtype, count=count, sep=sep, **kwargs)
        except ValueError as e:
            if sep == "" and "frombuffer" in str(e).lower():
                try:
                    return np.frombuffer(data, dtype=dtype, count=count)
                except Exception:
                    pass
            raise

    np.fromstring = _np_fromstring_compat  # type: ignore[assignment]
    np._meeting_scribe_fromstring_compat = True  # type: ignore[attr-defined]

import soundcard as sc

from audio.domain.formats import AudioFormat
from audio.infrastructure.sources.base import BaseSource


class WasapiLoopbackSource(BaseSource):
    """WASAPI loopback source implemented via soundcard."""

    def __init__(self, name: str, format: AudioFormat, device: Optional[Any] = None):
        super().__init__(name=name, format=format)
        self._device = device
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
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self.name}-loopback",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _pick_loopback_mic(self):
        candidates = list(sc.all_microphones(include_loopback=True))
        loopbacks = [m for m in candidates if bool(getattr(m, "isloopback", False))]
        search_space = loopbacks if loopbacks else candidates

        if not search_space:
            raise RuntimeError("No capture devices were found")

        if self._device is None:
            sp = sc.default_speaker()
            sp_id = getattr(sp, "id", None)
            for m in search_space:
                if getattr(m, "id", None) == sp_id:
                    return m
            return sc.get_microphone(sp.name, include_loopback=True)

        dev = self._device
        dev_name = str(dev).strip().lower()

        for m in search_space:
            if getattr(m, "id", None) == dev:
                return m

        for m in search_space:
            n = str(getattr(m, "name", "")).strip()
            if n and n.lower() == dev_name:
                return m

        for m in search_space:
            n = str(getattr(m, "name", "")).strip()
            if n and dev_name and dev_name in n.lower():
                return m

        try:
            return sc.get_microphone(dev, include_loopback=True)
        except Exception:
            return sc.get_microphone(str(dev), include_loopback=True)

    def _run(self) -> None:
        try:
            mic = self._pick_loopback_mic()

            with mic.recorder(
                samplerate=self.format.sample_rate,
                blocksize=self.format.blocksize,
            ) as rec:
                while not self._stop.is_set():
                    data = rec.record(numframes=self.format.blocksize)
                    if data is None:
                        continue

                    x = np.asarray(data, dtype=np.float32)
                    if self._on_audio is not None:
                        self._on_audio(self.name, x)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            self._set_last_error(err)
            cb = self._on_error
            if cb is not None:
                try:
                    cb(self.name, err)
                except Exception:
                    pass
