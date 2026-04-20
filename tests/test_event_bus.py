from __future__ import annotations

import unittest

from application.event_bus import QueuedEventBus
from application.event_types import CodexResultEvent, SourceErrorEvent, TypedEvent


class EventBusTests(unittest.TestCase):
    def test_dispatches_to_specific_and_catch_all_subscribers(self) -> None:
        bus = QueuedEventBus(maxsize=4)
        specific: list[TypedEvent] = []
        all_events: list[TypedEvent] = []

        bus.subscribe(CodexResultEvent, specific.append)
        bus.subscribe(None, all_events.append)

        bus.publish(CodexResultEvent(ok=True, profile="Fast", cmd="ANSWER", text="ok", dt_s=1.0))
        bus.publish(SourceErrorEvent(source="mic", error="boom"))

        self.assertEqual(bus.drain(limit=10), 2)
        self.assertEqual(len(specific), 1)
        self.assertEqual(len(all_events), 2)
        self.assertIsInstance(specific[0], CodexResultEvent)


if __name__ == "__main__":
    unittest.main()
