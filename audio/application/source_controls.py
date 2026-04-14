from __future__ import annotations

import time

from audio.application.source_state import SourceState
from audio.domain.formats import AudioFormat


def clear_source_buffers(state: SourceState) -> None:
    with state.buf_lock:
        state.buf.clear()
        state.delay_buf.clear()
        state.buffer_frames = 0


def reset_source_runtime_state(state: SourceState) -> None:
    clear_source_buffers(state)
    state.rms = 0.0
    state.last_ts = 0.0
    state.dropped_in_frames = 0
    state.missing_out_frames = 0


def set_source_enabled_state(state: SourceState, enabled: bool) -> None:
    new_value = bool(enabled)
    if state.enabled == new_value:
        return

    state.enabled = new_value
    clear_source_buffers(state)

    if not new_value:
        state.rms = 0.0
        state.last_ts = time.monotonic()


def set_source_delay(state: SourceState, fmt: AudioFormat, delay_ms: float) -> None:
    delay = max(0.0, float(delay_ms))
    state.delay_frames = int(round((delay / 1000.0) * fmt.sample_rate))
    with state.buf_lock:
        state.delay_buf.clear()
