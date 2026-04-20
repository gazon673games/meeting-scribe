from __future__ import annotations

import unittest

from application.job_state import (
    AssistantJobState,
    AssistantJobStateMachine,
    InvalidJobTransition,
    TranscriptionJobState,
    TranscriptionJobStateMachine,
)


class JobStateMachineTests(unittest.TestCase):
    def test_assistant_job_fallback_cycle(self) -> None:
        machine = AssistantJobStateMachine()

        machine.begin()
        self.assertEqual(machine.state, AssistantJobState.RUNNING)
        self.assertTrue(machine.is_busy)

        machine.begin_fallback()
        self.assertEqual(machine.state, AssistantJobState.FALLBACK)
        self.assertTrue(machine.is_fallback)

        machine.finish()
        self.assertEqual(machine.state, AssistantJobState.IDLE)
        self.assertFalse(machine.is_busy)

    def test_assistant_rejects_fallback_from_idle(self) -> None:
        machine = AssistantJobStateMachine()

        with self.assertRaises(InvalidJobTransition):
            machine.begin_fallback()

    def test_transcription_degraded_start_and_stop(self) -> None:
        machine = TranscriptionJobStateMachine()

        machine.begin_start()
        machine.begin_fallback()
        machine.finish_start(degraded=True)

        self.assertEqual(machine.state, TranscriptionJobState.RUNNING_DEGRADED)
        self.assertTrue(machine.is_running)
        self.assertTrue(machine.is_degraded)
        self.assertTrue(machine.can_stop)

        machine.begin_stop()
        self.assertEqual(machine.state, TranscriptionJobState.STOPPING)

        machine.finish_stop()
        self.assertEqual(machine.state, TranscriptionJobState.IDLE)

    def test_transcription_failed_can_start_again(self) -> None:
        machine = TranscriptionJobStateMachine()

        machine.begin_start()
        machine.fail_start()

        self.assertEqual(machine.state, TranscriptionJobState.FAILED)
        self.assertTrue(machine.can_start)


if __name__ == "__main__":
    unittest.main()
