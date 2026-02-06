# --- File: D:\work\own\voice2textTest\asr\vad.py ---
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EnergyVAD:
    """
    Dependency-free VAD for MVP.

    Operates on 16k mono float32 frames.

    IMPORTANT:
      - energy_threshold: lower -> more sensitive (more false positives on noise)
      - hangover_ms: keep speech after energy drops to avoid chopping
      - min_speech_ms: minimum duration required to accept a segment
    """
    frame_ms: int = 20
    sample_rate: int = 16000

    # For better recall on speech (quality mode often wants slightly lower thr)
    energy_threshold: float = 0.006

    # Quality mode: keep speech longer to avoid chopping words
    hangover_ms: int = 400

    # Quality mode: reject too short segments (reduce junk)
    min_speech_ms: int = 350

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

    def reset(self) -> None:
        self._hang = 0
        self._in_speech = False
        self._speech_frames = 0

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        if frame.size == 0:
            self._last_rms = 0.0
            return False

        e = float(np.sqrt(np.mean(frame.astype(np.float32, copy=False) ** 2)))
        self._last_rms = e
        self._last_thr = float(self.energy_threshold)

        speech_now = e >= float(self.energy_threshold)

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
