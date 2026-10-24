from __future__ import annotations

import queue

from application.event_types import event_to_record

RETAINED_EVENT_TYPES = frozenset(
    {
        'utterance',
        'streaming_final',
        'transcript_speaker_update',
        'error',
        'asr_stopped',
    }
)


def is_retained_event(event: object) -> bool:
    return event_to_record(event).get('type') in RETAINED_EVENT_TYPES


class AsrEventQueue(queue.Queue[object]):
    """Bound disposable updates; preserve ordered transcript and error events."""

    def put_nowait(self, item: object) -> None:
        if not is_retained_event(item):
            super().put_nowait(item)
            return
        with self.not_full:
            if self.maxsize > 0 and self._qsize() >= self.maxsize:
                for index, pending in enumerate(self.queue):
                    if not is_retained_event(pending):
                        del self.queue[index]
                        self.unfinished_tasks -= 1
                        break
            # Capacity is soft only for retained events. Blocking here can
            # deadlock startup/shutdown while the consumer holds a session lock.
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()
