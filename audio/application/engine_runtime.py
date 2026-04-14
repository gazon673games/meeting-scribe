from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Optional, Set, Tuple

from audio.application.tap_config import TapConfig
from audio.domain.types import TapMode


@dataclass
class EngineRuntimeState:
    tap_q: Optional["queue.Queue[dict]"] = None
    tap_queue_max: int = 200
    tap_mode: TapMode = "both"
    tap_sources_filter: Optional[Set[str]] = None
    tap_drop_threshold: float = 0.85
    tick_index: int = 0
    master_rms: float = 0.0
    master_last_ts: float = 0.0
    dropped_out_blocks: int = 0
    dropped_tap_blocks: int = 0
    autosync_enabled: bool = False
    autosync_ref: Optional[str] = None
    autosync_target: Optional[str] = None
    autosync_last_offset_ms: float = 0.0

    def set_tap_queue(self, tap_queue: Optional["queue.Queue[dict]"]) -> None:
        self.tap_q = tap_queue
        self.dropped_tap_blocks = 0

    def apply_tap_config(self, config: TapConfig) -> None:
        self.tap_mode = config.mode
        self.tap_sources_filter = config.sources_filter
        self.tap_drop_threshold = config.drop_threshold

    def enable_auto_sync(self, reference_source: str, target_source: str) -> None:
        self.autosync_enabled = True
        self.autosync_ref = reference_source
        self.autosync_target = target_source
        self.autosync_last_offset_ms = 0.0

    def disable_auto_sync(self) -> None:
        self.autosync_enabled = False
        self.autosync_ref = None
        self.autosync_target = None
        self.autosync_last_offset_ms = 0.0

    def reset_for_start(self) -> None:
        self.tick_index = 0
        self.dropped_out_blocks = 0
        self.dropped_tap_blocks = 0

    def reset_after_stop(self) -> None:
        self.master_rms = 0.0
        self.master_last_ts = 0.0
        self.dropped_out_blocks = 0
        self.dropped_tap_blocks = 0
        self.autosync_last_offset_ms = 0.0

    def next_mix_window(self, period_s: float) -> Tuple[float, float]:
        t_start = float(self.tick_index) * period_s
        self.tick_index += 1
        return t_start, t_start + period_s

    def record_master_metrics(self, master_rms: float, ts_mono: float) -> None:
        self.master_rms = float(master_rms)
        self.master_last_ts = float(ts_mono)

    def record_output_drop(self) -> None:
        self.dropped_out_blocks += 1

    def record_tap_drop(self) -> None:
        self.dropped_tap_blocks += 1
