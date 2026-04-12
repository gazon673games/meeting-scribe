from __future__ import annotations

from typing import List

import numpy as np

from audio.application.source_state import SourceState
from audio.domain.formats import AudioFormat
from audio.domain.ports import AudioFilter


def apply_filters(x: np.ndarray, fmt: AudioFormat, filters: List[AudioFilter]) -> np.ndarray:
    y = x
    for audio_filter in filters:
        y = audio_filter.process(y, fmt)
    return y


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    xf = x.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(xf * xf)))


def pad_or_crop_n(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[0] == n:
        return x.astype(np.float32, copy=False)
    if x.shape[0] > n:
        return x[:n].astype(np.float32, copy=False)
    pad = np.zeros((n - x.shape[0], x.shape[1]), dtype=np.float32)
    return np.vstack([x.astype(np.float32, copy=False), pad])


def channel_map_to_engine(x: np.ndarray, src_ch: int, eng_ch: int) -> np.ndarray:
    if src_ch == eng_ch:
        return x.astype(np.float32, copy=False)

    if src_ch == 1 and eng_ch == 2:
        return np.repeat(x, 2, axis=1).astype(np.float32, copy=False)

    if src_ch == 2 and eng_ch == 1:
        return x.mean(axis=1, keepdims=True).astype(np.float32, copy=False)

    n = x.shape[0]
    if x.shape[1] > eng_ch:
        return x[:, :eng_ch].astype(np.float32, copy=False)

    out = np.zeros((n, eng_ch), dtype=np.float32)
    out[:, : x.shape[1]] = x.astype(np.float32, copy=False)
    return out


def resample_to_engine_rate(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return x.astype(np.float32, copy=False)

    n = x.shape[0]
    if n == 0:
        return x.astype(np.float32, copy=False)

    duration = n / float(src_rate)
    dst_n = int(round(duration * dst_rate))
    if dst_n <= 0:
        return np.zeros((0, x.shape[1]), dtype=np.float32)

    src_t = np.linspace(0.0, duration, num=n, endpoint=False, dtype=np.float64)
    dst_t = np.linspace(0.0, duration, num=dst_n, endpoint=False, dtype=np.float64)

    out = np.zeros((dst_n, x.shape[1]), dtype=np.float32)
    for ch in range(x.shape[1]):
        out[:, ch] = np.interp(dst_t, src_t, x[:, ch].astype(np.float64, copy=False)).astype(np.float32, copy=False)
    return out


def apply_delay_block(state: SourceState, block: np.ndarray, blocksize: int, channels: int) -> np.ndarray:
    if state.delay_frames <= 0:
        return block

    blocks_delay = int(np.ceil(state.delay_frames / float(blocksize)))
    while len(state.delay_buf) < blocks_delay:
        state.delay_buf.append(np.zeros((blocksize, channels), dtype=np.float32))

    state.delay_buf.append(block)
    return state.delay_buf.popleft()
