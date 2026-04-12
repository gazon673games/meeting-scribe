from __future__ import annotations

import queue
import time
from typing import Callable

LogEvent = Callable[[dict], None]
EmitMetrics = Callable[[bool], None]


class TapIngestRuntime:
    def __init__(
        self,
        *,
        tap_queue: "queue.Queue[dict]",
        stop_event,
        mode: str,
        segmenter,
        log_event: LogEvent,
        emit_metrics: EmitMetrics,
    ) -> None:
        self._tap_q = tap_queue
        self._stop = stop_event
        self._mode = str(mode)
        self._segmenter = segmenter
        self._log_event = log_event
        self._emit_metrics = emit_metrics

    def run_safe(self) -> None:
        try:
            self.run()
        except Exception as e:
            self._log_event({"type": "error", "where": "ingest", "error": str(e), "ts": time.time()})

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                pkt = self._tap_q.get(timeout=0.2)
            except queue.Empty:
                continue

            self._segmenter.feed_packet(mode=self._mode, pkt=pkt)
            self._emit_metrics(False)
