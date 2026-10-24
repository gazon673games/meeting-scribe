from __future__ import annotations

import queue
import unittest
from unittest.mock import patch

from application.event_bus import QueuedEventBus
from application.event_types import (
    CodexResultEvent,
    SourceErrorEvent,
    TypedEvent,
)


class EventBusTests(unittest.TestCase):
    def test_expected_queue_races_drop_event_without_failing(self) -> None:
        bus = QueuedEventBus(maxsize=1)
        with (
            patch.object(
                bus._queue, 'put_nowait', side_effect=queue.Full
            ) as put,
            patch.object(bus._queue, 'get_nowait', side_effect=queue.Empty),
        ):
            bus.publish(SourceErrorEvent(source='mic', error='audio'))
        self.assertEqual(put.call_count, 2)

    def test_programming_errors_during_overflow_are_propagated(self) -> None:
        for phase in ('remove', 'retry'):
            with self.subTest(phase=phase):
                bus = QueuedEventBus(maxsize=1)
                remove_error = TypeError('bug') if phase == 'remove' else None
                with (
                    patch.object(
                        bus._queue,
                        'put_nowait',
                        side_effect=[queue.Full, TypeError('bug')],
                    ),
                    patch.object(
                        bus._queue, 'get_nowait', side_effect=remove_error
                    ),
                    self.assertRaisesRegex(TypeError, 'bug'),
                ):
                    bus.publish(SourceErrorEvent(source='mic', error='audio'))

    def test_dispatches_to_specific_and_catch_all_subscribers(self) -> None:
        bus = QueuedEventBus(maxsize=4)
        specific: list[TypedEvent] = []
        all_events: list[TypedEvent] = []

        bus.subscribe(CodexResultEvent, specific.append)
        bus.subscribe(None, all_events.append)

        bus.publish(
            CodexResultEvent(
                ok=True, profile='Fast', cmd='ANSWER', text='ok', dt_s=1.0
            )
        )
        bus.publish(SourceErrorEvent(source='mic', error='boom'))

        self.assertEqual(bus.drain(limit=10), 2)
        self.assertEqual(len(specific), 1)
        self.assertEqual(len(all_events), 2)
        self.assertIsInstance(specific[0], CodexResultEvent)

    def test_publish_drops_oldest_event_when_queue_is_full(self) -> None:
        bus = QueuedEventBus(maxsize=1)
        events: list[TypedEvent] = []
        bus.subscribe(None, events.append)

        bus.publish(SourceErrorEvent(source='old', error='first'))
        bus.publish(SourceErrorEvent(source='new', error='second'))

        self.assertEqual(bus.drain(limit=5), 1)
        self.assertEqual(events[0].source, 'new')


if __name__ == '__main__':
    unittest.main()
