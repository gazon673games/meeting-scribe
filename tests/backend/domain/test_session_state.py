from __future__ import annotations

import unittest

from session.domain.state import InvalidSessionTransition, SessionState, SessionStateMachine


class SessionStateMachineTests(unittest.TestCase):
    def test_start_and_stop_cycle(self) -> None:
        machine = SessionStateMachine()

        self.assertEqual(machine.state, SessionState.IDLE)
        self.assertTrue(machine.can_start)

        machine.begin_start()
        self.assertEqual(machine.state, SessionState.STARTING)
        self.assertFalse(machine.can_start)

        machine.finish_start()
        self.assertEqual(machine.state, SessionState.RUNNING)
        self.assertTrue(machine.can_stop)
        self.assertTrue(machine.is_running)

        machine.begin_stop()
        self.assertEqual(machine.state, SessionState.STOPPING)
        self.assertTrue(machine.is_stopping)

        machine.finish_stop()
        self.assertEqual(machine.state, SessionState.IDLE)

    def test_model_download_returns_to_idle_before_start(self) -> None:
        machine = SessionStateMachine()

        machine.begin_model_download()
        self.assertEqual(machine.state, SessionState.DOWNLOADING_MODEL)
        self.assertTrue(machine.is_downloading_model)

        machine.finish_model_download()
        self.assertEqual(machine.state, SessionState.IDLE)

    def test_offline_pass_blocks_start_until_finished(self) -> None:
        machine = SessionStateMachine()

        machine.begin_offline_pass()
        self.assertEqual(machine.state, SessionState.OFFLINE_PASS)
        self.assertTrue(machine.is_offline_pass)
        self.assertFalse(machine.can_start)

        machine.finish_offline_pass()
        self.assertEqual(machine.state, SessionState.IDLE)
        self.assertTrue(machine.can_start)

    def test_invalid_transition_raises(self) -> None:
        machine = SessionStateMachine()

        with self.assertRaises(InvalidSessionTransition):
            machine.begin_stop()

        machine.begin_start()
        with self.assertRaises(InvalidSessionTransition):
            machine.begin_offline_pass()


if __name__ == "__main__":
    unittest.main()
