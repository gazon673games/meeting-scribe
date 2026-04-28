from __future__ import annotations

import unittest

from application.codex_config import DEFAULT_CODEX_PROXY, CodexProfile, parse_codex_settings
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

    def test_codex_proxy_can_be_disabled_explicitly(self) -> None:
        self.assertEqual(parse_codex_settings({}).proxy, DEFAULT_CODEX_PROXY)
        self.assertEqual(parse_codex_settings({"proxy": ""}).proxy, "")

    def test_codex_profiles_keep_provider_and_model_settings(self) -> None:
        settings = parse_codex_settings(
            {
                "profiles": [
                    {
                        "id": "deep",
                        "label": "Deep",
                        "provider": "codex",
                        "model": "gpt-5.3-codex",
                        "reasoning_effort": "high",
                    }
                ]
            }
        )

        self.assertEqual(settings.profiles[0].provider_id, "codex")
        self.assertEqual(settings.profiles[0].model, "gpt-5.3-codex")
        self.assertEqual(settings.profiles[0].reasoning_effort, "high")


if __name__ == "__main__":
    unittest.main()
