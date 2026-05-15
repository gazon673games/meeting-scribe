from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from interface.assistant_controller import AssistantController
from interface.jsonl_bridge import JsonLineBridge
from interface.session_controller import HeadlessSessionController
from settings.infrastructure.json_config_repository import JsonConfigRepository
from tests.helpers.electron_interface_fakes import (
    _FakeAssistantService,
    _FakeAudioRuntimeFactory,
    _FakeAudioSourceFactory,
    _FakeWavRecorderFactory,
)


class ElectronInterfaceAssistantBridgeTests(unittest.TestCase):
    def test_assistant_controller_uses_session_transcript_context(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "codex": {
                        "enabled": True,
                        "selected_profile": "fast",
                        "profiles": [{"id": "fast", "label": "Fast", "prompt": "help"}],
                    }
                }
            )
            session = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
            )
            session._transcript_lines.append({"ts": 1.0, "stream": "mic", "text": "question"})  # noqa: SLF001
            events: list[tuple[str, dict]] = []
            assistant = AssistantController(
                project_root=root,
                config_repository=repository,
                assistant_service=_FakeAssistantService(),  # type: ignore[arg-type]
                session_controller=session,
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )

            snapshot = assistant.invoke({"requestText": "reply"})
            for _ in range(20):
                if any(kind == "assistant_result" for kind, _ in events):
                    break
                time.sleep(0.02)

            self.assertIn("profiles", snapshot)
            self.assertTrue(any(kind == "assistant_result" for kind, _ in events))
            self.assertIn("question", assistant.snapshot()["lastResponse"]["text"])

    def test_assistant_controller_allows_empty_transcript_for_direct_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "codex": {
                        "enabled": True,
                        "selected_profile": "fast",
                        "profiles": [{"id": "fast", "label": "Fast", "prompt": "help"}],
                    }
                }
            )
            session = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
            )
            events: list[tuple[str, dict]] = []
            assistant = AssistantController(
                project_root=root,
                config_repository=repository,
                assistant_service=_FakeAssistantService(),  # type: ignore[arg-type]
                session_controller=session,
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )

            snapshot = assistant.invoke({"requestText": "reply"})
            for _ in range(20):
                if any(kind == "assistant_result" for kind, _ in events):
                    break
                time.sleep(0.02)

            self.assertIn("profiles", snapshot)
            self.assertTrue(any(kind == "assistant_result" for kind, _ in events))
            self.assertEqual(assistant.snapshot()["lastRequest"]["contextLabel"], "no transcript")

    def test_assistant_controller_emits_local_model_status_events(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "codex": {
                        "enabled": True,
                        "selected_profile": "qwen",
                        "profiles": [
                            {
                                "id": "qwen",
                                "label": "qwen",
                                "provider": "openai_local",
                                "model": "local-model",
                                "base_url": "http://127.0.0.1:1234/v1",
                            }
                        ],
                    }
                }
            )
            events: list[tuple[str, dict]] = []
            assistant = AssistantController(
                project_root=root,
                config_repository=repository,
                assistant_service=_FakeAssistantService(),  # type: ignore[arg-type]
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )

            def fake_start(profile, project_root, emit_event):  # noqa: ANN001
                emit_event(
                    {
                        "type": "local_llm_status",
                        "profileId": profile.id,
                        "state": "running",
                        "message": "ready",
                    }
                )
                return {"started": True, "profileId": profile.id}

            with patch("infrastructure.local_llm.start_local_llm_async", side_effect=fake_start):
                result = assistant.start_local_model({"profileId": "qwen"})

            self.assertTrue(result["started"])
            self.assertEqual(len(events), 1)
            kind, payload = events[0]
            self.assertEqual(kind, "local_llm_status")
            self.assertEqual(payload["profileId"], "qwen")
            self.assertEqual(payload["state"], "running")
            self.assertEqual(payload["message"], "ready")
            self.assertIn("ts", payload)

    def test_jsonl_bridge_wraps_success_and_errors(self) -> None:
        stdin = io.StringIO(
            "\n".join(
                [
                    json.dumps({"id": "1", "method": "ping", "params": {"value": 42}}),
                    json.dumps({"id": "2", "method": "missing"}),
                    "",
                ]
            )
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        def handler(method, params):  # noqa: ANN001
            if method == "ping":
                return {"pong": True, "params": params}
            raise KeyError(method)

        JsonLineBridge(handler, stdin=stdin, stdout=stdout, stderr=stderr).serve_forever()
        messages = [json.loads(line) for line in stdout.getvalue().splitlines()]

        self.assertEqual(messages[0]["event"]["type"], "backend_ready")
        self.assertTrue(messages[1]["ok"])
        self.assertEqual(messages[1]["result"]["params"]["value"], 42)
        self.assertFalse(messages[2]["ok"])
        self.assertEqual(messages[2]["error"]["type"], "KeyError")

    def test_jsonl_bridge_preserves_unicode_payloads(self) -> None:
        text = "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440"
        reply_prefix = "\u041e\u0442\u0432\u0435\u0442:"
        stdin = io.StringIO(
            json.dumps(
                {"id": "ru", "method": "echo", "params": {"text": text}},
                ensure_ascii=False,
            )
            + "\n"
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        def handler(method, params):  # noqa: ANN001
            self.assertEqual(method, "echo")
            return {"reply": f"{reply_prefix} {params['text']}"}

        JsonLineBridge(handler, stdin=stdin, stdout=stdout, stderr=stderr).serve_forever()
        raw_output = stdout.getvalue()
        messages = [json.loads(line) for line in raw_output.splitlines()]

        self.assertEqual(messages[1]["result"]["reply"], f"{reply_prefix} {text}")
        self.assertIn(text, raw_output)

