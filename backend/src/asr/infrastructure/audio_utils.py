from __future__ import annotations

import numpy as np


def _make_decim3_kernel() -> np.ndarray:
    # Windowed-sinc lowpass FIR for 3x decimation (48 kHz → 16 kHz).
    # Cutoff 7500 Hz, 31 taps, Hann window.  Computed once at import time.
    n = 31
    fc = 7500.0 / 24000.0  # normalised to input Nyquist (48 000 / 2)
    half = (n - 1) / 2.0
    t = np.arange(n, dtype=np.float64) - half
    h = 2.0 * fc * np.sinc(2.0 * fc * t)
    h = h * np.hanning(n)
    h = (h / h.sum()).astype(np.float32)
    return h


_DECIM3_KERNEL: np.ndarray = _make_decim3_kernel()


def stereo_to_mono(x: np.ndarray) -> np.ndarray:
    """
    More robust stereo->mono.

    Problem with naive mean():
      - if one channel is near-silent (common with some loopback/mic configs),
        mean reduces amplitude ~2x and can push speech below VAD threshold.
    Strategy:
      - if mono already -> return
      - if 2ch -> compute per-channel RMS on this block:
          - if one channel dominates (ratio >= 2.0), use dominant channel
          - else use average
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)

    if x.shape[1] == 1:
        return x[:, 0].astype(np.float32, copy=False)

    # generic: take first 2 channels for analysis, but average all if >2
    ch0 = x[:, 0].astype(np.float32, copy=False)
    ch1 = x[:, 1].astype(np.float32, copy=False)

    r0 = float(np.sqrt(np.mean(ch0 * ch0))) if ch0.size else 0.0
    r1 = float(np.sqrt(np.mean(ch1 * ch1))) if ch1.size else 0.0

    # avoid div by zero
    hi = max(r0, r1)
    lo = max(1e-12, min(r0, r1))

    if hi / lo >= 2.0:
        # pick stronger channel
        return (ch0 if r0 >= r1 else ch1).astype(np.float32, copy=False)

    # otherwise average all channels
    return x.astype(np.float32, copy=False).mean(axis=1)


def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    x: (n,) float32
    returns (m,) float32
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate:
        return x.astype(np.float32, copy=False)
    n = int(x.shape[0])
    if n <= 1:
        return x.astype(np.float32, copy=False)

    if src_rate == 48000 and dst_rate == 16000:
        m = int(round(n / 3.0))
        if m <= 0:
            return np.zeros((0,), dtype=np.float32)
        # Anti-aliasing FIR before decimation prevents aliasing of 8–24 kHz
        # content (sibilants, fricatives) into the 0–8 kHz band Whisper sees.
        filtered = np.convolve(x, _DECIM3_KERNEL, mode='same')
        return filtered[: m * 3 : 3].astype(np.float32, copy=True)

    dur = n / float(src_rate)
    m = int(round(dur * dst_rate))
    if m <= 0:
        return np.zeros((0,), dtype=np.float32)

    src_t = np.linspace(0.0, dur, num=n, endpoint=False, dtype=np.float64)
    dst_t = np.linspace(0.0, dur, num=m, endpoint=False, dtype=np.float64)
    y = np.interp(dst_t, src_t, x.astype(np.float64, copy=False)).astype(np.float32, copy=False)
    return y
