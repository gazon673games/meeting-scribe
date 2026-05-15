from __future__ import annotations

import queue
import unittest

from application.event_types import (
    AsrMetricsEvent,
    CodexFallbackStartedEvent,
    CodexResultEvent,
    EventType,
    SourceErrorEvent,
    TranscriptSpeakerUpdateEvent,
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
        event = event_from_record(
            {
                "type": "utterance",
                "text": "hello",
                "stream": "mic",
                "speaker": "Me",
                "t_start": 1.0,
                "t_end": 1.4,
                "overload": True,
                "ts": 1.5,
            }
        )

        self.assertIsInstance(event, UtteranceEvent)
        assert isinstance(event, UtteranceEvent)
        self.assertEqual(event.event_type, EventType.UTTERANCE)
        self.assertEqual(event.text, "hello")
        self.assertEqual(event.stream, "mic")
        self.assertEqual(event.speaker, "Me")
        self.assertEqual(event.t_start, 1.0)
        self.assertEqual(event.t_end, 1.4)
        self.assertTrue(event.overload)
        self.assertEqual(event.ts, 1.5)

    def test_decodes_transcript_speaker_update_event(self) -> None:
        event = event_from_record(
            {
                "type": "transcript_speaker_update",
                "line_id": "desktop:1000:2000",
                "stream": "desktop",
                "speaker": "Remote S1",
                "t_start": 1.0,
                "t_end": 2.0,
                "confidence": 0.82,
                "source": "diarization",
                "ts": 3.0,
            }
        )

        self.assertIsInstance(event, TranscriptSpeakerUpdateEvent)
        assert isinstance(event, TranscriptSpeakerUpdateEvent)
        self.assertEqual(event.speaker, "Remote S1")
        self.assertEqual(event.confidence, 0.82)
        self.assertEqual(event_to_record(event)["type"], "transcript_speaker_update")

    def test_typed_event_round_trips_to_record(self) -> None:
        event = CodexResultEvent(
            ok=False,
            profile="Fast",
            cmd="ANSWER",
            text="network down",
            dt_s=1.25,
            provider="codex",
            model="gpt-5.3-codex",
            error_code="network_error",
            retryable=True,
            suggestion="Check proxy",
            ts=10.0,
        )
        record = event_to_record(event)

        self.assertEqual(record["type"], "codex_result")
        self.assertEqual(record["profile"], "Fast")
        self.assertEqual(record["cmd"], "ANSWER")
        self.assertEqual(record["dt_s"], 1.25)
        self.assertEqual(record["provider"], "codex")
        self.assertEqual(record["model"], "gpt-5.3-codex")
        self.assertEqual(record["error_code"], "network_error")

        decoded = event_from_record(record)
        self.assertIsInstance(decoded, CodexResultEvent)
        self.assertTrue(decoded.retryable)  # type: ignore[attr-defined]

    def test_asr_started_event_keeps_runtime_diagnostics(self) -> None:
        event = event_from_record(
            {
                "type": "asr_started",
                "model": "large-v3",
                "mode": "split",
                "language": "ru",
                "device": "cuda",
                "compute_type": "float16",
                "cpu_threads": 0,
                "num_workers": 2,
                "beam_size": 5,
                "overload_strategy": "drop_old",
                "ts": 2.0,
            }
        )

        self.assertEqual(event_to_record(event)["device"], "cuda")
        self.assertEqual(event_to_record(event)["compute_type"], "float16")
        self.assertEqual(event_to_record(event)["num_workers"], 2)
        self.assertEqual(event_to_record(event)["beam_size"], 5)

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

    def test_asr_publisher_publishes_speaker_updates_and_debug_events(self) -> None:
        q: queue.Queue[object] = queue.Queue()
        logger = _Logger()
        publisher = ASREventPublisher(logger=logger, event_queue=q)

        publisher.log({"type": "transcript_speaker_update", "stream": "desktop", "speaker": "Remote S1", "ts": 12.0})
        publisher.log({"type": "diar_segment_processing", "stream": "desktop", "ts": 13.0})

        self.assertIsInstance(q.get_nowait(), TranscriptSpeakerUpdateEvent)
        self.assertEqual(q.get_nowait().as_record()["type"], "diar_segment_processing")

    def test_explicit_ui_events_are_typed(self) -> None:
        self.assertEqual(SourceErrorEvent(source="mic", error="boom").event_type, EventType.SOURCE_ERROR)
        self.assertEqual(
            CodexFallbackStartedEvent(profile="Fast", cmd="ANSWER", reason="timeout").event_type,
            EventType.CODEX_FALLBACK_STARTED,
        )


if __name__ == "__main__":
    unittest.main()
