from __future__ import annotations

import time
from collections import deque
from typing import Deque


class ASRMetrics:
    def __init__(self, *, latency_window: int, emit_interval_s: float) -> None:
        self._lat_samples: Deque[float] = deque(maxlen=max(20, int(latency_window)))
        self._emit_interval_s = max(0.25, float(emit_interval_s))
        self.reset()

    def reset(self) -> None:
        self.seg_dropped_total = 0
        self.seg_skipped_total = 0
        self.last_lag_s = 0.0
        self._last_emit_ts = 0.0
        self._lat_samples.clear()

    def record_segment_dropped(self, count: int = 1) -> None:
        self.seg_dropped_total += max(0, int(count))

    def record_segments_skipped(self, count: int) -> None:
        self.seg_skipped_total += max(0, int(count))

    def record_latency(self, *, asr_latency_s: float, total_lag_s: float) -> None:
        self.last_lag_s = float(total_lag_s)
        self._lat_samples.append(float(asr_latency_s))

    def build_event(
        self,
        *,
        force: bool,
        seg_qsize: int,
        overload_active: bool,
        overload_strategy: str,
        hard_overload: bool,
    ) -> dict | None:
        now = time.time()
        if not force and (now - float(self._last_emit_ts)) < self._emit_interval_s:
            return None
        self._last_emit_ts = now

        lat_list = list(self._lat_samples)
        avg_lat = float(sum(lat_list) / max(1, len(lat_list))) if lat_list else 0.0
        p95 = 0.0
        if lat_list:
            lat_sorted = sorted(lat_list)
            idx = int(round(0.95 * (len(lat_sorted) - 1)))
            idx = max(0, min(len(lat_sorted) - 1, idx))
            p95 = float(lat_sorted[idx])

        return {
            "type": "asr_metrics",
            "seg_dropped_total": int(self.seg_dropped_total),
            "seg_skipped_total": int(self.seg_skipped_total),
            "avg_latency_s": float(avg_lat),
            "p95_latency_s": float(p95),
            "lag_s": float(self.last_lag_s),
            "seg_qsize": int(seg_qsize),
            "overload": bool(overload_active),
            "overload_strategy": str(overload_strategy),
            "hard_overload": bool(hard_overload),
            "ts": now,
        }
