from __future__ import annotations

import queue
from typing import Any, Iterable, Set


PUBLIC_EVENT_TYPES = {
    "utterance",
    "asr_overload",
    "segment_dropped",
    "segment_skipped_overload",
    "asr_metrics",
    "asr_init_start",
    "asr_started",
    "asr_init_ok",
    "error",
    "asr_stopped",
}


class ASREventPublisher:
    def __init__(self, *, logger: Any, event_queue: "queue.Queue[dict] | None", public_types: Iterable[str] | None = None):
        self._logger = logger
        self._event_q = event_queue
        self._public_types: Set[str] = set(public_types or PUBLIC_EVENT_TYPES)

    def log(self, rec: dict) -> None:
        try:
            self._logger.write(rec)
        except Exception:
            pass
        self.publish(rec)

    def publish(self, rec: dict) -> None:
        if not self._event_q:
            return
        typ = str(rec.get("type", ""))
        if typ not in self._public_types:
            return
        try:
            self._event_q.put_nowait(rec)
        except queue.Full:
            pass
