# --- File: D:\work\own\voice2textTest\asr\vad.py ---
from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Tuple
from collections import deque
import numpy as np


@dataclass
class VADDecision:
    speech: bool
    energy_rms: float
    thr: float
    noise_rms: float
    band_ratio: float
    voiced: float


class EnergyVAD:
    """
    Step 3 VAD: energy + speech-band ratio + voicedness + better segmentation helpers.

    Long-run fixes:
      - cache FFT window, freqs, and band mask (no per-frame allocations)
      - compute voicedness less aggressively (only near threshold and/or every N frames)
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        energy_threshold: float = 0.006,
        adaptive: bool = True,
        noise_mult: float = 3.0,
        noise_alpha: float = 0.05,
        hangover_ms: int = 400,
        min_speech_ms: int = 350,

        band_ratio_min: float = 0.35,
        voiced_min: float = 0.12,
        band_ratio_weight: float = 0.55,
        voiced_weight: float = 0.45,

        pre_speech_ms: int = 120,
        min_end_silence_ms: int = 220,

        # NEW (perf knobs)
        voiced_every_n_frames: int = 2,      # compute voicedness once per N frames
        voiced_only_near_thr: bool = True,   # compute voicedness only when energy is close to threshold
        near_thr_ratio: float = 0.70,        # "near threshold" gate (uses previous thr)
    ):
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_len = int(round(self.sample_rate * self.frame_ms / 1000.0))

        self._min_thr = float(energy_threshold)
        self._adaptive = bool(adaptive)
        self._noise_mult = float(noise_mult)
        self._noise_alpha = float(noise_alpha)

        self._hangover_frames = max(0, int(round(hangover_ms / self.frame_ms)))
        self._min_speech_frames = max(1, int(round(min_speech_ms / self.frame_ms)))

        self._pre_frames = max(0, int(round(pre_speech_ms / self.frame_ms)))
        self._min_end_silence_frames = max(0, int(round(min_end_silence_ms / self.frame_ms)))

        self._band_ratio_min = float(band_ratio_min)
        self._voiced_min = float(voiced_min)
        self._band_ratio_weight = float(band_ratio_weight)
        self._voiced_weight = float(voiced_weight)

        self._noise_rms = 0.0
        self._thr = self._min_thr

        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._hangover_left = 0

        self._last_rms = 0.0
        self._last_band_ratio = 0.0
        self._last_voiced = 0.0

        # ring buffer for pre-roll frames
        self._prebuf: Deque[np.ndarray] = deque(maxlen=max(1, self._pre_frames) if self._pre_frames > 0 else 1)

        # ===== NEW: cached FFT helpers =====
        self._win = np.hanning(self.frame_len).astype(np.float32, copy=False)
        self._freqs = np.fft.rfftfreq(self.frame_len, d=1.0 / float(self.sample_rate))
        lo, hi = 300.0, 3400.0
        self._band_mask = (self._freqs >= lo) & (self._freqs <= hi)

        # ===== NEW: voicedness throttling =====
        self._voiced_every_n = max(1, int(voiced_every_n_frames))
        self._voiced_only_near_thr = bool(voiced_only_near_thr)
        self._near_thr_ratio = float(near_thr_ratio)
        if self._near_thr_ratio < 0.1:
            self._near_thr_ratio = 0.1
        if self._near_thr_ratio > 1.0:
            self._near_thr_ratio = 1.0
        self._frame_index = 0

    def reset(self) -> None:
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._hangover_left = 0
        self._prebuf.clear()
        self._frame_index = 0

    def last_rms(self) -> float:
        return float(self._last_rms)

    def last_threshold(self) -> float:
        return float(self._thr)

    def noise_rms(self) -> float:
        return float(self._noise_rms)

    def last_band_ratio(self) -> float:
        return float(self._last_band_ratio)

    def last_voiced(self) -> float:
        return float(self._last_voiced)

    def speech_long_enough(self) -> bool:
        return int(self._speech_run) >= int(self._min_speech_frames)

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        xf = x.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(xf * xf) + 1e-12))

    @staticmethod
    def _remove_dc(x: np.ndarray) -> np.ndarray:
        xf = x.astype(np.float32, copy=False)
        return xf - float(np.mean(xf))

    def _band_ratio(self, x: np.ndarray) -> float:
        xf = self._remove_dc(x)
        xw = xf * self._win
        spec = np.fft.rfft(xw)
        pwr = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32, copy=False)

        total = float(np.sum(pwr) + 1e-12)
        band = float(np.sum(pwr[self._band_mask]) + 1e-12)
        return float(band / total)

    def _voicedness(self, x: np.ndarray) -> float:
        xf = self._remove_dc(x)
        denom = float(np.dot(xf, xf) + 1e-12)
        if denom <= 1e-10:
            return 0.0

        min_f, max_f = 80.0, 400.0
        min_lag = int(self.sample_rate / max_f)
        max_lag = int(self.sample_rate / min_f)
        max_lag = min(max_lag, xf.size - 1)
        if max_lag <= min_lag + 2:
            return 0.0

        best = 0.0
        for lag in range(min_lag, max_lag):
            v = float(np.dot(xf[:-lag], xf[lag:]) / denom)
            if v > best:
                best = v
        return float(max(0.0, min(1.0, best)))

    def _update_noise(self, rms: float, band_ratio: float, voiced: float) -> None:
        if not self._adaptive:
            self._thr = self._min_thr
            return

        not_speech_like = (band_ratio < (self._band_ratio_min * 0.85)) and (voiced < (self._voiced_min * 0.85))
        if not_speech_like:
            if self._noise_rms <= 0.0:
                self._noise_rms = float(rms)
            else:
                a = float(self._noise_alpha)
                self._noise_rms = (1.0 - a) * float(self._noise_rms) + a * float(rms)

        self._thr = max(self._min_thr, float(self._noise_rms) * float(self._noise_mult))

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        if x.size != self.frame_len:
            if x.size < self.frame_len:
                pad = np.zeros((self.frame_len - x.size,), dtype=np.float32)
                x = np.concatenate([x, pad])
            else:
                x = x[: self.frame_len]

        self._frame_index += 1

        rms = self._rms(x)
        band_ratio = self._band_ratio(x)

        # NEW: voicedness throttling
        compute_voiced = True
        if self._voiced_only_near_thr:
            # use previous threshold as a cheap gate
            compute_voiced = bool(rms >= (float(self._thr) * float(self._near_thr_ratio)))

        if compute_voiced and (self._frame_index % self._voiced_every_n == 0):
            voiced = self._voicedness(x)
            self._last_voiced = float(voiced)
        else:
            voiced = float(self._last_voiced)

        self._last_rms = float(rms)
        self._last_band_ratio = float(band_ratio)

        self._update_noise(rms, band_ratio, voiced)

        energy_ok = rms >= float(self._thr)

        band_ok = band_ratio >= float(self._band_ratio_min)
        voiced_ok = voiced >= float(self._voiced_min)

        score = (float(self._band_ratio_weight) * (1.0 if band_ok else 0.0)) + (
            float(self._voiced_weight) * (1.0 if voiced_ok else 0.0)
        )

        near_energy = rms >= (0.85 * float(self._thr))
        speech_like = score >= 0.60

        speech = bool(energy_ok or (near_energy and speech_like))

        if speech:
            self._speech_run += 1
            self._silence_run = 0
            self._hangover_left = int(self._hangover_frames)
            self._in_speech = True
        else:
            self._silence_run += 1
            if self._hangover_left > 0:
                self._hangover_left -= 1
                speech = True
                self._in_speech = True
            else:
                if self._in_speech and self._silence_run <= int(self._min_end_silence_frames):
                    speech = True
                    self._in_speech = True
                else:
                    self._in_speech = False

        if self._pre_frames > 0:
            self._prebuf.append(x.copy())

        return bool(speech)

    def pop_preroll(self) -> Tuple[np.ndarray, int]:
        if self._pre_frames <= 0 or not self._prebuf:
            return np.zeros((0,), dtype=np.float32), 0
        frames = list(self._prebuf)
        self._prebuf.clear()
        cat = np.concatenate(frames).astype(np.float32, copy=False)
        return cat, len(frames)
