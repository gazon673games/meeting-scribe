from __future__ import annotations

import unittest

from assistant.domain.aggregate import AssistantJobAggregate
from assistant.domain.events import AssistantFallbackStarted, AssistantRequestFinished, AssistantRequestStarted
from session.domain.aggregate import SessionAggregate
from session.domain.events import OfflinePassFinished, SessionStartFailed, SessionStarted, SessionStartRequested
from session.domain.state import SessionStateMachine
from transcription.domain.aggregate import TranscriptionJobAggregate
from transcription.domain.events import TranscriptionFallbackStarted, TranscriptionStartFailed, TranscriptionStarted


class DomainAggregateTests(unittest.TestCase):
    def test_session_aggregate_records_domain_events(self) -> None:
        session = SessionAggregate()
        session.begin_model_download("large-v3")
        self.assertTrue(session.is_downloading_model)
        session.finish_model_download("large-v3")

        session.begin_start(source_count=2, asr_enabled=True, profile="Realtime", language="ru")
        session.finish_start(asr_running=True, wav_recording=False)

        events = session.pull_domain_events()
        self.assertIsInstance(events[2], SessionStartRequested)
        self.assertIsInstance(events[3], SessionStarted)
        self.assertEqual(events[2].source_count, 2)
        self.assertTrue(events[3].asr_running)
        self.assertEqual(session.pull_domain_events(), [])

    def test_session_aggregate_records_failed_start_and_offline_pass(self) -> None:
        session = SessionAggregate()
        self.assertEqual(session.domain_events, ())

        session.begin_start()
        session.fail_start("bad source")
        session.begin_offline_pass("session.wav")
        self.assertTrue(session.is_offline_pass)
        session.finish_offline_pass(out_txt="out.txt", error="")

        events = session.pull_domain_events()
        self.assertIsInstance(events[1], SessionStartFailed)
        self.assertIsInstance(events[3], OfflinePassFinished)

        state = SessionStateMachine()
        state.begin_start()
        state.fail_start()
        self.assertTrue(state.can_start)

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
        self.assertTrue(job.can_start)
        self.assertEqual(job.state.value, "idle")

        job.begin_start(model_name="large-v3", mode="split", language="ru")
        job.begin_fallback(attempt_label="fast-fallback", model_name="small", reason="cuda error")
        job.finish_start(degraded=True, attempt_label="fast-fallback")

        events = job.pull_domain_events()
        self.assertIsInstance(events[1], TranscriptionFallbackStarted)
        self.assertIsInstance(events[2], TranscriptionStarted)
        self.assertTrue(events[2].degraded)
        self.assertTrue(job.is_running)
        self.assertTrue(job.is_degraded)

        job.begin_stop()
        job.finish_stop()
        job.begin_start()
        job.fail_start("failed")
        failed_events = job.pull_domain_events()
        self.assertIsInstance(failed_events[-1], TranscriptionStartFailed)


if __name__ == "__main__":
    unittest.main()
