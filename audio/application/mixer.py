from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio.application.dsp import (
    apply_delay_block,
    apply_filters,
    channel_map_to_engine,
    pad_or_crop_n,
    resample_to_engine_rate,
    rms,
)
from audio.application.source_state import SourceState
from audio.domain import AudioFormat


@dataclass(frozen=True)
class RenderedSourceBlock:
    block: np.ndarray
    active: bool


def normalize_source_frame(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame)
    if x.ndim == 1:
        x = x[:, None]
    return np.array(x, dtype=np.float32, copy=True)


def enqueue_source_frame(state: SourceState, frame: np.ndarray, *, max_buffer_blocks: int) -> None:
    with state.buf_lock:
        state.buf.append(frame)
        state.buffer_frames += int(len(frame))
        while len(state.buf) > int(max_buffer_blocks):
            dropped = state.buf.popleft()
            state.dropped_in_frames += int(len(dropped))
            state.buffer_frames = max(0, int(state.buffer_frames) - int(len(dropped)))


def render_source_block(
    *,
    state: SourceState,
    engine_format: AudioFormat,
    ts_mono: float,
    active_rms_eps: float,
) -> RenderedSourceBlock:
    src_fmt = state.src.get_format()
    state.src_rate = int(src_fmt.sample_rate)

    if not state.enabled:
        block_eng = np.zeros((engine_format.blocksize, engine_format.channels), dtype=np.float32)
        state.rms = 0.0
        state.last_ts = ts_mono
        return RenderedSourceBlock(block=block_eng, active=False)

    raw = None
    with state.buf_lock:
        if state.buf:
            raw = state.buf.popleft()
            state.buffer_frames = max(0, int(state.buffer_frames) - int(len(raw)))

    if raw is None:
        block_src = np.zeros((engine_format.blocksize, max(1, src_fmt.channels)), dtype=np.float32)
        state.missing_out_frames += engine_format.blocksize
    else:
        block_src = raw
        if block_src.ndim == 1:
            block_src = block_src[:, None]

    if block_src.shape[0] < engine_format.blocksize:
        pad = np.zeros((engine_format.blocksize - block_src.shape[0], block_src.shape[1]), dtype=np.float32)
        block_src = np.vstack([block_src, pad])
        if raw is not None:
            state.missing_out_frames += engine_format.blocksize - raw.shape[0]
    elif block_src.shape[0] > engine_format.blocksize:
        block_src = block_src[: engine_format.blocksize]

    try:
        block_src = apply_filters(block_src, src_fmt, state.src.get_filters())
    except Exception:
        block_src = np.zeros((engine_format.blocksize, max(1, src_fmt.channels)), dtype=np.float32)
        state.missing_out_frames += engine_format.blocksize

    block_src = resample_to_engine_rate(block_src, int(src_fmt.sample_rate), int(engine_format.sample_rate))

    if block_src.ndim == 1:
        block_src = block_src[:, None]
    block_src = pad_or_crop_n(block_src, engine_format.blocksize)

    block_eng = channel_map_to_engine(block_src, int(src_fmt.channels), engine_format.channels)

    if state.delay_frames > 0:
        block_eng = apply_delay_block(state, block_eng, engine_format.blocksize, engine_format.channels)

    state.rms = rms(block_eng)
    state.last_ts = ts_mono

    return RenderedSourceBlock(block=block_eng, active=float(state.rms) > float(active_rms_eps))
