# --- File: D:\work\own\voice2textTest\audio\sources\wasapi_loopback.py ---
from __future__ import annotations

from typing import Callable, Optional
import threading

import numpy as np
import soundcard as sc

from audio.engine import AudioFormat
from audio.sources.base import BaseSource


class WasapiLoopbackSource(BaseSource):
    """
    Захват системного звука (что играет на динамиках/наушниках) через SoundCard loopback.
    SoundCard 0.4.x: записываем НЕ из Speaker, а из Microphone с include_loopback=True.

    Важно: канал-маппинг и blocksize выравнивание делает engine, тут отдаем "как пришло".
    """

    def __init__(self, name: str, format: AudioFormat, device: Optional[str] = None):
        """
        device:
          - None: loopback для default speaker
          - str: подстрока для поиска loopback-микрофона (например "Realtek", "Headphones", "Scarlett")
        """
        super().__init__(name=name, format=format)
        self._device = device
        self._on_audio: Optional[Callable[[str, np.ndarray], None]] = None

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self, on_audio: Callable[[str, np.ndarray], None]) -> None:
        self._on_audio = on_audio
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
        if self._device is None:
            sp = sc.default_speaker()
            return sc.get_microphone(sp.name, include_loopback=True)

        candidates = sc.all_microphones(include_loopback=True)
        for m in candidates:
            if self._device.lower() in m.name.lower():
                return m

        return sc.get_microphone(self._device, include_loopback=True)

    def _run(self) -> None:
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
                # do not channel-map here; engine will normalize
                if self._on_audio is not None:
                    self._on_audio(self.name, x)
