from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from application.codex_config import CodexProfile
from assistant.application.provider import AssistantExecutionSettings, AssistantProviderRequest
from infrastructure.local_llm import OllamaLocalLlmRunner, OpenAICompatibleLocalLlmRunner


class _Response:
    status = 200

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.closed = False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def close(self) -> None:
        self.closed = True


class _Opener:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.request = None
        self.timeout = None

    def open(self, request, timeout=None):  # noqa: ANN001
        self.request = request
        self.timeout = timeout
        return self.response


def _settings(root: Path, profile: CodexProfile) -> AssistantExecutionSettings:
    return AssistantExecutionSettings(
        command_tokens=["codex"],
        path_hints=[],
        proxy="",
        timeout_s=30,
        project_root=root,
        profile=profile,
        profiles=[profile],
    )


class LocalLlmProviderTests(unittest.TestCase):
    def test_ollama_status_uses_profile_base_url_and_lists_models(self) -> None:
        profile = CodexProfile(
            id="ollama",
            label="Ollama",
            prompt="",
            provider_id="ollama",
            model="llama3.2",
            base_url="http://127.0.0.1:11434",
        )
        opener = _Opener(_Response({"models": [{"name": "llama3.2"}]}))

        with patch("infrastructure.local_llm.urllib.request.build_opener", return_value=opener):
            status = OllamaLocalLlmRunner().status(_settings(Path("."), profile))

        self.assertTrue(status.available)
        self.assertEqual(status.models, ["llama3.2"])
        self.assertEqual(opener.request.full_url, "http://127.0.0.1:11434/api/tags")

    def test_ollama_run_posts_generate_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            profile = CodexProfile(
                id="ollama",
                label="Ollama",
                prompt="",
                provider_id="ollama",
                model="llama3.2",
                base_url="http://127.0.0.1:11434",
                temperature=0.2,
                max_tokens=64,
            )
            opener = _Opener(_Response({"response": "local answer"}))
            request = AssistantProviderRequest(
                prompt="question",
                profile=profile,
                original_cmd="ANSWER",
                project_root=root,
                settings=_settings(root, profile),
            )

            with patch("infrastructure.local_llm.urllib.request.build_opener", return_value=opener):
                result = OllamaLocalLlmRunner().run(request)

        body = json.loads(opener.request.data.decode("utf-8"))
        self.assertTrue(result.ok)
        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.text, "local answer")
        self.assertEqual(opener.request.full_url, "http://127.0.0.1:11434/api/generate")
        self.assertEqual(body["model"], "llama3.2")
        self.assertEqual(body["options"]["temperature"], 0.2)
        self.assertEqual(body["options"]["num_predict"], 64)

    def test_openai_compatible_run_posts_chat_completion_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            profile = CodexProfile(
                id="lmstudio",
                label="LM Studio",
                prompt="",
                provider_id="openai_local",
                model="local-model",
                base_url="http://127.0.0.1:1234/v1",
                api_key="secret",
            )
            opener = _Opener(_Response({"choices": [{"message": {"content": "chat answer"}}]}))
            request = AssistantProviderRequest(
                prompt="question",
                profile=profile,
                original_cmd="ANSWER",
                project_root=root,
                settings=_settings(root, profile),
            )

            with patch("infrastructure.local_llm.urllib.request.build_opener", return_value=opener):
                result = OpenAICompatibleLocalLlmRunner().run(request)

        body = json.loads(opener.request.data.decode("utf-8"))
        self.assertTrue(result.ok)
        self.assertEqual(result.provider, "openai_local")
        self.assertEqual(result.text, "chat answer")
        self.assertEqual(opener.request.full_url, "http://127.0.0.1:1234/v1/chat/completions")
        self.assertEqual(body["model"], "local-model")
        self.assertEqual(body["messages"][0]["content"], "question")
        self.assertEqual(opener.request.get_header("Authorization"), "Bearer secret")


if __name__ == "__main__":
    unittest.main()
