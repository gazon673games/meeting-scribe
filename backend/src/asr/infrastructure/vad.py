from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Tuple

import numpy as np

# ── speech decision thresholds ─────────────────────────────────────────────
_SCORE_THRESHOLD: float = 0.60       # weighted band+voiced score to count as speech-like
_NEAR_ENERGY_RATIO: float = 0.85     # fraction of threshold for "near energy" gate

# ── voicedness autocorrelation pitch range ─────────────────────────────────
_PITCH_MIN_HZ: float = 80.0
_PITCH_MAX_HZ: float = 400.0


@dataclass(frozen=True)
class VadFeatureConfig:
    band_ratio_min: float = 0.35
    voiced_min: float = 0.12
    band_ratio_weight: float = 0.55
    voiced_weight: float = 0.45
    voiced_every_n_frames: int = 2
    voiced_only_near_thr: bool = True
    near_thr_ratio: float = 0.70


@dataclass(frozen=True)
class VadHangoverConfig:
    hangover_ms: int = 400
    min_speech_ms: int = 350
    pre_speech_ms: int = 120
    min_end_silence_ms: int = 220


@dataclass
class _SpeechStateMachine:
    hangover_frames: int
    min_speech_frames: int
    min_end_silence_frames: int

    in_speech: bool = field(default=False, init=False)
    speech_run: int = field(default=0, init=False)
    silence_run: int = field(default=0, init=False)
    hangover_left: int = field(default=0, init=False)

    def reset(self) -> None:
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.hangover_left = 0

    def update(self, raw_speech: bool) -> bool:
        if raw_speech:
            self.speech_run += 1
            self.silence_run = 0
            self.hangover_left = self.hangover_frames
            self.in_speech = True
            return True

        self.silence_run += 1
        if self.hangover_left > 0:
            self.hangover_left -= 1
            self.in_speech = True
            return True
        if self.in_speech and self.silence_run <= self.min_end_silence_frames:
            self.in_speech = True
            return True
        self.in_speech = False
        return False

    def speech_long_enough(self) -> bool:
        return self.speech_run >= self.min_speech_frames


class EnergyVAD:
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
        voiced_every_n_frames: int = 2,
        voiced_only_near_thr: bool = True,
        near_thr_ratio: float = 0.70,
    ):
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_len = int(round(self.sample_rate * self.frame_ms / 1000.0))

        self._min_thr = float(energy_threshold)
        self._adaptive = bool(adaptive)
        self._noise_mult = float(noise_mult)
        self._noise_alpha = float(noise_alpha)
        self._noise_rms = 0.0
        self._thr = self._min_thr

        self._features = VadFeatureConfig(
            band_ratio_min=band_ratio_min,
            voiced_min=voiced_min,
            band_ratio_weight=band_ratio_weight,
            voiced_weight=voiced_weight,
            voiced_every_n_frames=max(1, int(voiced_every_n_frames)),
            voiced_only_near_thr=bool(voiced_only_near_thr),
            near_thr_ratio=max(0.1, min(1.0, float(near_thr_ratio))),
        )
        self._sm = _SpeechStateMachine(
            hangover_frames=max(0, int(round(hangover_ms / self.frame_ms))),
            min_speech_frames=max(1, int(round(min_speech_ms / self.frame_ms))),
            min_end_silence_frames=max(0, int(round(min_end_silence_ms / self.frame_ms))),
        )
        self._pre_frames = max(0, int(round(pre_speech_ms / self.frame_ms)))
        self._prebuf: Deque[np.ndarray] = deque(maxlen=max(1, self._pre_frames) if self._pre_frames > 0 else 1)

        # cached FFT helpers — avoids per-frame allocation
        self._win = np.hanning(self.frame_len).astype(np.float32, copy=False)
        self._freqs = np.fft.rfftfreq(self.frame_len, d=1.0 / float(self.sample_rate))
        self._band_mask = (self._freqs >= 300.0) & (self._freqs <= 3400.0)

        self._last_rms = 0.0
        self._last_band_ratio = 0.0
        self._last_voiced = 0.0
        self._frame_index = 0

    # ── public API ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._sm.reset()
        self._prebuf.clear()
        self._frame_index = 0

    def speech_long_enough(self) -> bool:
        return self._sm.speech_long_enough()

    def last_rms(self) -> float:       return float(self._last_rms)
    def last_threshold(self) -> float: return float(self._thr)
    def noise_rms(self) -> float:      return float(self._noise_rms)
    def last_band_ratio(self) -> float: return float(self._last_band_ratio)
    def last_voiced(self) -> float:    return float(self._last_voiced)

    def is_speech_frame(self, frame: np.ndarray) -> bool:
        x = self._normalize_frame(frame)
        rms, band_ratio, voiced = self._extract_features(x)
        self._update_noise(rms, band_ratio, voiced)
        raw_speech = self._decide_raw_speech(rms, band_ratio, voiced)
        speech = self._sm.update(raw_speech)
        if self._pre_frames > 0:
            self._prebuf.append(x.copy())
        return speech

    def pop_preroll(self) -> Tuple[np.ndarray, int]:
        if self._pre_frames <= 0 or not self._prebuf:
            return np.zeros((0,), dtype=np.float32), 0
        frames = list(self._prebuf)
        self._prebuf.clear()
        return np.concatenate(frames).astype(np.float32, copy=False), len(frames)

    # ── frame processing ────────────────────────────────────────────────────

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        if x.size < self.frame_len:
            x = np.concatenate([x, np.zeros((self.frame_len - x.size,), dtype=np.float32)])
        elif x.size > self.frame_len:
            x = x[: self.frame_len]
        return x

    def _extract_features(self, x: np.ndarray) -> Tuple[float, float, float]:
        self._frame_index += 1
        rms = self._rms(x)
        if rms < self._thr * min(_NEAR_ENERGY_RATIO, self._features.near_thr_ratio):
            self._last_rms = rms
            self._last_band_ratio = 0.0
            self._last_voiced = 0.0
            return rms, 0.0, 0.0
        band_ratio = self._band_ratio(x)
        voiced = self._compute_voiced_throttled(x, rms)
        self._last_rms = rms
        self._last_band_ratio = band_ratio
        return rms, band_ratio, voiced

    def _compute_voiced_throttled(self, x: np.ndarray, rms: float) -> float:
        feat = self._features
        if feat.voiced_only_near_thr and rms < self._thr * feat.near_thr_ratio:
            return self._last_voiced
        if self._frame_index % feat.voiced_every_n_frames != 0:
            return self._last_voiced
        voiced = self._voicedness(x)
        self._last_voiced = voiced
        return voiced

    def _decide_raw_speech(self, rms: float, band_ratio: float, voiced: float) -> bool:
        feat = self._features
        energy_ok = rms >= self._thr
        score = (feat.band_ratio_weight * (1.0 if band_ratio >= feat.band_ratio_min else 0.0)
                 + feat.voiced_weight   * (1.0 if voiced    >= feat.voiced_min        else 0.0))
        near_energy = rms >= _NEAR_ENERGY_RATIO * self._thr
        return bool(energy_ok or (near_energy and score >= _SCORE_THRESHOLD))

    # ── signal analysis ─────────────────────────────────────────────────────

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        xf = x.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(xf * xf) + 1e-12))

    @staticmethod
    def _remove_dc(x: np.ndarray) -> np.ndarray:
        xf = x.astype(np.float32, copy=False)
        return xf - float(np.mean(xf))

    def _band_ratio(self, x: np.ndarray) -> float:
        xw = self._remove_dc(x) * self._win
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
        min_lag = int(self.sample_rate / _PITCH_MAX_HZ)
        max_lag = min(int(self.sample_rate / _PITCH_MIN_HZ), xf.size - 1)
        if max_lag <= min_lag + 2:
            return 0.0
        best = max(float(np.dot(xf[:-lag], xf[lag:]) / denom) for lag in range(min_lag, max_lag))
        return float(max(0.0, min(1.0, best)))

    # ── noise adaptation ────────────────────────────────────────────────────

    def _update_noise(self, rms: float, band_ratio: float, voiced: float) -> None:
        if not self._adaptive:
            self._thr = self._min_thr
            return
        feat = self._features
        not_speech_like = (band_ratio < feat.band_ratio_min * 0.85) and (voiced < feat.voiced_min * 0.85)
        if not_speech_like:
            if self._noise_rms <= 0.0:
                self._noise_rms = rms
            else:
                a = self._noise_alpha
                self._noise_rms = (1.0 - a) * self._noise_rms + a * rms
        self._thr = max(self._min_thr, self._noise_rms * self._noise_mult)
