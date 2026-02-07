# --- File: D:\work\own\voice2textTest\asr\vad.py ---
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EnergyVAD:
    """
    Dependency-free VAD for MVP with optional noise-adaptive threshold.

    Operates on 16k mono float32 frames.

    Tuning:
      - energy_threshold: base floor (lower -> more sensitive)
      - adaptive: if enabled, threshold becomes max(base, noise_rms * noise_mult)
      - hangover_ms: keep speech after energy drops
      - min_speech_ms: minimum duration to accept a segment
    """
    frame_ms: int = 20
    sample_rate: int = 16000

    energy_threshold: float = 0.006

    hangover_ms: int = 400
    min_speech_ms: int = 350

    # NEW: adaptive threshold
    adaptive: bool = True
    noise_mult: float = 3.0          # threshold = max(base, noise_rms * noise_mult)
    noise_alpha: float = 0.05        # EMA update speed for noise floor
    noise_update_only_on_silence: bool = True

    def __post_init__(self) -> None:
        self.frame_len = int(self.sample_rate * self.frame_ms / 1000)

        self.hangover_frames = max(0, int(self.hangover_ms / self.frame_ms))
        self.min_speech_frames = max(1, int(self.min_speech_ms / self.frame_ms))

        self._hang = 0
        self._in_speech = False
        self._speech_frames = 0

        # debug
        self._last_rms = 0.0
        self._last_thr = float(self.energy_threshold)

        # NEW: noise floor estimate (EMA of RMS)
        self._noise_rms = 0.0

    def reset(self) -> None:
        self._hang = 0
        self._in_speech = False
        self._speech_frames = 0

    def _calc_thr(self) -> float:
        base = float(self.energy_threshold)
        if not self.adaptive:
            return base
        # if noise not established yet, keep base
        if self._noise_rms <= 1e-9:
            return base
        return max(base, float(self._noise_rms) * float(self.noise_mult))

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        if frame.size == 0:
            self._last_rms = 0.0
            self._last_thr = self._calc_thr()
            return False

        fr = frame.astype(np.float32, copy=False)
        e = float(np.sqrt(np.mean(fr * fr)))
        self._last_rms = e

        thr = self._calc_thr()
        self._last_thr = float(thr)

        speech_now = e >= thr

        # NEW: update noise floor
        if self.adaptive:
            can_update = True
            if self.noise_update_only_on_silence and (speech_now or self._in_speech):
                can_update = False
            if can_update:
                a = float(self.noise_alpha)
                if self._noise_rms <= 1e-9:
                    self._noise_rms = e
                else:
                    self._noise_rms = (1.0 - a) * float(self._noise_rms) + a * e

        if speech_now:
            self._in_speech = True
            self._hang = self.hangover_frames
            self._speech_frames += 1
            return True

        if self._in_speech:
            if self._hang > 0:
                self._hang -= 1
                self._speech_frames += 1
                return True
            else:
                self._in_speech = False

        return False

    def speech_long_enough(self) -> bool:
        return self._speech_frames >= self.min_speech_frames

    # debug hooks
    def last_rms(self) -> float:
        return float(self._last_rms)

    def last_threshold(self) -> float:
        return float(self._last_thr)

    def noise_rms(self) -> float:
        return float(self._noise_rms)
