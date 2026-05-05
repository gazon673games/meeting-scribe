from __future__ import annotations

import queue
import time
from typing import Callable

from asr.application.ports import StopSignalPort
from asr.application.segmentation import AudioSegmenterPort

LogEvent = Callable[[dict], None]


class TapIngestRuntime:
    def __init__(
        self,
        *,
        tap_queue: "queue.Queue[dict]",
        stop_event: StopSignalPort,
        mode: str,
        segmenter: AudioSegmenterPort,
        log_event: LogEvent,
    ) -> None:
        self._tap_q = tap_queue
        self._stop = stop_event
        self._mode = str(mode)
        self._segmenter = segmenter
        self._log_event = log_event

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
