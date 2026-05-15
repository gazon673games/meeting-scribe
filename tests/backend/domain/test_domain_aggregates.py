from __future__ import annotations

import unittest

from assistant.domain.aggregate import AssistantJobAggregate
from assistant.domain.events import AssistantFallbackStarted, AssistantRequestFinished, AssistantRequestStarted
from session.domain.aggregate import SessionAggregate
from session.domain.events import SessionStarted, SessionStartRequested
from transcription.domain.aggregate import TranscriptionJobAggregate
from transcription.domain.events import TranscriptionFallbackStarted, TranscriptionStarted


class DomainAggregateTests(unittest.TestCase):
    def test_session_aggregate_records_domain_events(self) -> None:
        session = SessionAggregate()

        session.begin_start(source_count=2, asr_enabled=True, profile="Realtime", language="ru")
        session.finish_start(asr_running=True, wav_recording=False)

        events = session.pull_domain_events()
        self.assertIsInstance(events[0], SessionStartRequested)
        self.assertIsInstance(events[1], SessionStarted)
        self.assertEqual(events[0].source_count, 2)
        self.assertTrue(events[1].asr_running)
        self.assertEqual(session.pull_domain_events(), [])

    def test_assistant_aggregate_records_fallback_lifecycle(self) -> None:
        job = AssistantJobAggregate()

        job.begin(profile="Fast", source_label="answer")
        job.begin_fallback(profile="Fast", reason="timeout")
        job.finish(profile="Fast", ok=True, elapsed_s=12.5)

        events = job.pull_domain_events()
        self.assertIsInstance(events[0], AssistantRequestStarted)
        self.assertIsInstance(events[1], AssistantFallbackStarted)
        self.assertIsInstance(events[2], AssistantRequestFinished)
        self.assertEqual(events[1].reason, "timeout")

    def test_transcription_aggregate_records_degraded_start(self) -> None:
        job = TranscriptionJobAggregate()

        job.begin_start(model_name="large-v3", mode="split", language="ru")
        job.begin_fallback(attempt_label="fast-fallback", model_name="small", reason="cuda error")
        job.finish_start(degraded=True, attempt_label="fast-fallback")

        events = job.pull_domain_events()
        self.assertIsInstance(events[1], TranscriptionFallbackStarted)
        self.assertIsInstance(events[2], TranscriptionStarted)
        self.assertTrue(events[2].degraded)


if __name__ == "__main__":
    unittest.main()
