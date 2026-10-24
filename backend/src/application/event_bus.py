from __future__ import annotations

import logging
import queue
from collections import defaultdict
from collections.abc import Callable

from application.event_types import TypedEvent, event_from_record

EventHandler = Callable[[TypedEvent], None]


_logger = logging.getLogger(__name__)


class QueuedEventBus:
    def __init__(self, *, maxsize: int = 200) -> None:
        self._queue: queue.Queue[TypedEvent] = queue.Queue(maxsize=int(maxsize))
        self._handlers: defaultdict[
            type[TypedEvent] | None, list[EventHandler]
        ] = defaultdict(list)

    def subscribe(
        self, event_cls: type[TypedEvent] | None, handler: EventHandler
    ) -> None:
        self._handlers[event_cls].append(handler)

    def publish(self, raw_event: object) -> None:
        event = event_from_record(raw_event)
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                _logger.debug('Event queue drained concurrently', exc_info=True)
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                _logger.debug(
                    'Event queue refilled concurrently; event dropped',
                    exc_info=True,
                )

    def drain(self, *, limit: int = 100) -> int:
        count = 0
        while count < int(limit):
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            count += 1
            self._dispatch(event)
        return count

    def _dispatch(self, event: TypedEvent) -> None:
        for handler in list(self._handlers.get(type(event), [])):
            handler(event)
        for handler in list(self._handlers.get(None, [])):
            handler(event)
