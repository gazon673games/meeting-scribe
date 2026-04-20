from __future__ import annotations

import unittest

from application.codex_config import CodexProfile
from application.commands import InvokeAssistantCommand, StartSessionCommand, StopSessionCommand, SwitchProfileCommand


class CommandObjectTests(unittest.TestCase):
    def test_session_and_profile_commands_are_plain_intents(self) -> None:
        start = StartSessionCommand(
            source_count=2,
            asr_enabled=True,
            model_name="medium",
            profile="Realtime",
            language="ru",
        )
        stop = StopSessionCommand(run_offline_pass=False, wait=True)
        switch = SwitchProfileCommand(profile="Quality")

        self.assertEqual(start.model_name, "medium")
        self.assertTrue(start.asr_enabled)
        self.assertFalse(stop.run_offline_pass)
        self.assertTrue(stop.wait)
        self.assertEqual(switch.profile, "Quality")

    def test_invoke_assistant_command_carries_context_source(self) -> None:
        profile = CodexProfile(id="fast", label="Fast", prompt="", reasoning_effort="low")
        command = InvokeAssistantCommand(
            profile=profile,
            request_text="ANSWER",
            source_label="answer",
            context_source="transcript",
            context_label="current transcript",
            context_text="current words",
            max_log_chars=4000,
            timeout_s=35,
            fallback_max_log_chars=2000,
            fallback_timeout_s=20,
        )

        self.assertEqual(command.profile.id, "fast")
        self.assertEqual(command.context_source, "transcript")
        self.assertEqual(command.context_text, "current words")
        self.assertEqual(command.fallback_max_log_chars, 2000)


if __name__ == "__main__":
    unittest.main()
