from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from application.codex_config import CodexProfile
from assistant.application.provider import AssistantExecutionSettings, AssistantProviderRequest
from infrastructure import local_llm
from infrastructure.local_llm import LocalLlmError, OllamaLocalLlmRunner, OpenAICompatibleLocalLlmRunner
from infrastructure.local_llm_parts import result_utils, runtime_server
from infrastructure.local_llm_parts.errors import LocalLlmError as PartsLocalLlmError
from infrastructure.local_llm_parts.profile_utils import openai_models


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
    def tearDown(self) -> None:
        local_llm._SERVER_PROCESSES.clear()

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

        with patch("infrastructure.local_llm_parts.http_client.urllib.request.build_opener", return_value=opener):
            status = OllamaLocalLlmRunner().status(_settings(Path("."), profile))

        self.assertTrue(status.available)
        self.assertEqual(status.models, ["llama3.2"])
        self.assertEqual(opener.request.full_url, "http://127.0.0.1:11434/api/tags")

    def test_ollama_ping_and_openai_status_report_reachable_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            ollama_profile = CodexProfile(
                id="ollama",
                label="Ollama",
                prompt="",
                provider_id="ollama",
                model="llama3.2",
                base_url="http://127.0.0.1:11434",
            )
            opener = _Opener(_Response({"models": []}))
            with patch("infrastructure.local_llm_parts.http_client.urllib.request.build_opener", return_value=opener):
                ping = OllamaLocalLlmRunner().ping(_settings(root, ollama_profile))
            self.assertTrue(ping.ok)
            self.assertEqual(opener.request.full_url, "http://127.0.0.1:11434/api/tags")

            openai_profile = CodexProfile(
                id="lmstudio",
                label="LM Studio",
                prompt="",
                provider_id="openai_local",
                model="local-model",
                base_url="http://127.0.0.1:1234/v1",
            )
            opener = _Opener(_Response({"data": [{"id": "local-model"}]}))
            with patch("infrastructure.local_llm_parts.http_client.urllib.request.build_opener", return_value=opener):
                status = OpenAICompatibleLocalLlmRunner().status(_settings(root, openai_profile))
            self.assertTrue(status.available)
            self.assertEqual(status.models, ["local-model"])

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

            with patch("infrastructure.local_llm_parts.http_client.urllib.request.build_opener", return_value=opener):
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

            with patch("infrastructure.local_llm_parts.http_client.urllib.request.build_opener", return_value=opener):
                result = OpenAICompatibleLocalLlmRunner().run(request)

        body = json.loads(opener.request.data.decode("utf-8"))
        self.assertTrue(result.ok)
        self.assertEqual(result.provider, "openai_local")
        self.assertEqual(result.text, "chat answer")
        self.assertEqual(opener.request.full_url, "http://127.0.0.1:1234/v1/chat/completions")
        self.assertEqual(body["model"], "local-model")
        self.assertEqual(body["messages"][0]["content"], "question")
        self.assertEqual(opener.request.get_header("Authorization"), "Bearer secret")

    def test_openai_ping_only_checks_configured_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            server = root / ".local" / "llama_cpp" / "b1" / "llama-server.exe"
            server.parent.mkdir(parents=True)
            server.write_bytes(b"exe")
            model = root / "models" / "llm" / "qwen" / "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"gguf")
            profile = CodexProfile(
                id="local",
                label="Local",
                prompt="",
                provider_id="openai_local",
                model="Qwen2.5-Coder-7B-Instruct-Q4_K_M",
                base_url="http://127.0.0.1:1234/v1",
            )
            process = Mock()
            process.poll.return_value = None
            process.returncode = None

            with (
                patch(
                    "infrastructure.local_llm._request_json",
                    side_effect=LocalLlmError("local_llm_unavailable", "refused"),
                ),
                patch("infrastructure.local_llm_parts.runtime_server.subprocess.Popen", return_value=process) as popen,
            ):
                result = OpenAICompatibleLocalLlmRunner().ping(_settings(root, profile))

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "local_llm_unavailable")
        popen.assert_not_called()

    def test_status_ping_result_helpers_and_top_level_start_stop_wrappers_delegate(self) -> None:
        error = PartsLocalLlmError("offline", "Endpoint is offline", retryable=False, suggestion="start it", status_code=503)

        self.assertEqual(error.as_provider_error().code, "offline")
        self.assertFalse(result_utils.status_error("ollama", "Ollama", error, "http://x").retryable)
        self.assertTrue(result_utils.ping_ok("ollama", "Ollama", "ok").ok)
        self.assertEqual(result_utils.model_required("Ollama").code, "model_required")
        self.assertEqual(
            result_utils.error_result(CodexProfile("p", "Profile", ""), "cmd", "ollama", "m", error.as_provider_error(), 0).error_code,
            "offline",
        )
        self.assertEqual(openai_models({"data": [{"id": "a"}, {"id": ""}, object()]}), ["a"])

        profile = CodexProfile(id="local", label="Local", prompt="", provider_id="openai_local")
        with (
            patch("infrastructure.local_llm._start_local_llm_async", return_value={"started": True}) as start,
            patch("infrastructure.local_llm._stop_local_llm", return_value={"stopped": True}) as stop,
        ):
            self.assertTrue(local_llm.start_local_llm_async(profile, Path("."), lambda event: None)["started"])
            self.assertTrue(local_llm.stop_local_llm(profile)["stopped"])
        start.assert_called_once()
        stop.assert_called_once()

    def test_runtime_server_start_async_emits_synchronous_statuses_with_fake_thread(self) -> None:
        class ImmediateThread:
            def __init__(self, *, target, **kwargs):  # noqa: ANN001, ANN003
                self.target = target

            def start(self) -> None:
                self.target()

        events: list[dict] = []
        profile = SimpleNamespace(id="local", label="Local", base_url="http://127.0.0.1:1234/v1")

        with (
            patch("infrastructure.local_llm_parts.runtime_server.threading.Thread", ImmediateThread),
            patch("infrastructure.local_llm_parts.runtime_server.ensure_openai_local_runtime", return_value="ready"),
        ):
            result = runtime_server.start_local_llm_async(
                profile,
                Path("."),
                events.append,
                default_base_url="http://127.0.0.1:1234/v1",
            )

        self.assertTrue(result["started"])
        self.assertEqual([event["state"] for event in events], ["starting", "running"])
        with patch("infrastructure.local_llm_parts.runtime_server.socket.create_connection", side_effect=OSError):
            self.assertFalse(runtime_server._localhost_port_accepts_connection("127.0.0.1", 9))


if __name__ == "__main__":
    unittest.main()
