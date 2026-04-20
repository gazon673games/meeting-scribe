from __future__ import annotations

import queue
import unittest

from application.event_types import (
    AsrMetricsEvent,
    CodexFallbackStartedEvent,
    CodexResultEvent,
    EventType,
    SourceErrorEvent,
    UtteranceEvent,
    event_from_record,
    event_to_record,
)
from asr.application.events import ASREventPublisher


class _Logger:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, rec: dict) -> None:
        self.records.append(dict(rec))


class EventTypesTests(unittest.TestCase):
    def test_decodes_legacy_dict_to_typed_event(self) -> None:
        event = event_from_record({"type": "utterance", "text": "hello", "stream": "mic", "overload": True, "ts": 1.5})

        self.assertIsInstance(event, UtteranceEvent)
        assert isinstance(event, UtteranceEvent)
        self.assertEqual(event.event_type, EventType.UTTERANCE)
        self.assertEqual(event.text, "hello")
        self.assertEqual(event.stream, "mic")
        self.assertTrue(event.overload)
        self.assertEqual(event.ts, 1.5)

    def test_typed_event_round_trips_to_record(self) -> None:
        event = CodexResultEvent(ok=True, profile="Fast", cmd="ANSWER", text="done", dt_s=1.25, ts=10.0)
        record = event_to_record(event)

        self.assertEqual(record["type"], "codex_result")
        self.assertEqual(record["profile"], "Fast")
        self.assertEqual(record["cmd"], "ANSWER")
        self.assertEqual(record["dt_s"], 1.25)

        decoded = event_from_record(record)
        self.assertIsInstance(decoded, CodexResultEvent)

    def test_asr_publisher_logs_dict_and_publishes_typed_event(self) -> None:
        q: queue.Queue[object] = queue.Queue()
        logger = _Logger()
        publisher = ASREventPublisher(logger=logger, event_queue=q)

        publisher.log({"type": "asr_metrics", "seg_dropped_total": 2, "seg_skipped_total": 1, "ts": 12.0})

        self.assertEqual(logger.records[0]["type"], "asr_metrics")
        event = q.get_nowait()
        self.assertIsInstance(event, AsrMetricsEvent)
        assert isinstance(event, AsrMetricsEvent)
        self.assertEqual(event.seg_dropped_total, 2)
        self.assertEqual(event.seg_skipped_total, 1)

    def test_explicit_ui_events_are_typed(self) -> None:
        self.assertEqual(SourceErrorEvent(source="mic", error="boom").event_type, EventType.SOURCE_ERROR)
        self.assertEqual(
            CodexFallbackStartedEvent(profile="Fast", cmd="ANSWER", reason="timeout").event_type,
            EventType.CODEX_FALLBACK_STARTED,
        )


if __name__ == "__main__":
    unittest.main()
