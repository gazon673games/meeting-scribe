from __future__ import annotations

import unittest

from application.command_bus import CommandDispatcher
from application.commands import StopSessionCommand


class CommandDispatcherTests(unittest.TestCase):
    def test_dispatches_registered_command_handler(self) -> None:
        dispatcher = CommandDispatcher()
        calls: list[StopSessionCommand] = []

        dispatcher.register(StopSessionCommand, calls.append)
        command = StopSessionCommand(run_offline_pass=False, wait=True)

        dispatcher.dispatch(command)

        self.assertEqual(calls, [command])

    def test_missing_handler_is_explicit(self) -> None:
        dispatcher = CommandDispatcher()

        with self.assertRaises(KeyError):
            dispatcher.dispatch(StopSessionCommand())


if __name__ == "__main__":
    unittest.main()
