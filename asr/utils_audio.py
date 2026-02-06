# --- File: D:\work\own\voice2textTest\asr\utils_audio.py ---
from __future__ import annotations

import numpy as np


def stereo_to_mono(x: np.ndarray) -> np.ndarray:
    """
    x: (n, ch)
    returns (n,) float32
    """
    if x.ndim == 1:
        return x.astype(np.float32, copy=False)
    if x.shape[1] == 1:
        return x[:, 0].astype(np.float32, copy=False)
    return x.mean(axis=1).astype(np.float32, copy=False)


def resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    x: (n,) float32
    returns (m,) float32
    """
    if src_rate == dst_rate:
        return x.astype(np.float32, copy=False)
    n = int(x.shape[0])
    if n <= 1:
        return x.astype(np.float32, copy=False)

    dur = n / float(src_rate)
    m = int(round(dur * dst_rate))
    if m <= 0:
        return np.zeros((0,), dtype=np.float32)

    src_t = np.linspace(0.0, dur, num=n, endpoint=False, dtype=np.float64)
    dst_t = np.linspace(0.0, dur, num=m, endpoint=False, dtype=np.float64)
    y = np.interp(dst_t, src_t, x.astype(np.float64, copy=False)).astype(np.float32, copy=False)
    return y
