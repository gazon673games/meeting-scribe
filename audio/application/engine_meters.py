from __future__ import annotations

import queue
from typing import Dict, Optional, Set

from audio.application.source_state import SourceState
from audio.domain import AudioFormat, TapMode


def build_meter_snapshot(
    *,
    fmt: AudioFormat,
    state: Dict[str, SourceState],
    master_rms: float,
    master_last_ts: float,
    dropped_out_blocks: int,
    dropped_tap_blocks: int,
    tap_q: Optional["queue.Queue[dict]"],
    tap_mode: TapMode,
    tap_sources_filter: Optional[Set[str]],
    tap_drop_threshold: float,
    tap_queue_max: int,
    autosync_enabled: bool,
    autosync_ref: Optional[str],
    autosync_target: Optional[str],
    autosync_last_offset_ms: float,
) -> dict:
    return {
        "sources": {
            name: {
                "enabled": bool(src_state.enabled),
                "rms": float(src_state.rms),
                "last_ts": float(src_state.last_ts),
                "dropped_in_frames": int(src_state.dropped_in_frames),
                "missing_out_frames": int(src_state.missing_out_frames),
                "buffer_frames": int(src_state.buffer_frames),
                "delay_ms": float(src_state.delay_frames) * 1000.0 / float(fmt.sample_rate),
                "src_rate": int(src_state.src_rate),
            }
            for name, src_state in state.items()
        },
        "master": {
            "rms": float(master_rms),
            "last_ts": float(master_last_ts),
        },
        "drops": {
            "dropped_out_blocks": int(dropped_out_blocks),
            "dropped_tap_blocks": int(dropped_tap_blocks),
        },
        "tap": {
            "enabled": bool(tap_q is not None),
            "mode": str(tap_mode),
            "filter": sorted(list(tap_sources_filter)) if tap_sources_filter else None,
            "drop_threshold": float(tap_drop_threshold),
            "tap_queue_max": int(tap_queue_max),
        },
        "autosync": {
            "enabled": bool(autosync_enabled),
            "ref": autosync_ref,
            "target": autosync_target,
            "last_offset_ms": float(autosync_last_offset_ms),
        },
    }
