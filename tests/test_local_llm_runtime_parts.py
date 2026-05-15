from __future__ import annotations

import tempfile
import unittest
import urllib.error
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from assistant.application.provider import AssistantExecutionSettings
from infrastructure.local_llm_parts import runtime_server
from infrastructure.local_llm_parts.errors import LocalLlmError
from infrastructure.local_llm_parts.http_utils import http_error_body, http_suggestion
from infrastructure.local_llm_parts.runtime_discovery import find_gguf_model, find_llama_server


class _HttpError(urllib.error.HTTPError):
    def __init__(self, body: bytes) -> None:
        super().__init__("http://local", 500, "err", hdrs=None, fp=BytesIO(body))


class LocalLlmRuntimePartsTests(unittest.TestCase):
    def tearDown(self) -> None:
        runtime_server.SERVER_PROCESSES.clear()

    def test_http_error_helpers_extract_body_and_choose_status_suggestions(self) -> None:
        self.assertEqual(http_error_body(_HttpError(b" nope \n")), "nope")
        self.assertIn("model name", http_suggestion(404))
        self.assertIn("API key", http_suggestion(401))
        self.assertIn("busy", http_suggestion(429))
        self.assertIn("server logs", http_suggestion(503))
        self.assertIn("profile", http_suggestion(400))

    def test_find_llama_server_prefers_env_and_reports_missing_executable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            server = root / "bin" / "llama-server.exe"
            server.parent.mkdir()
            server.write_bytes(b"exe")

            with patch.dict("os.environ", {"LLAMA_SERVER_EXE": str(server)}):
                self.assertEqual(find_llama_server(root), server.resolve())

            with patch.dict("os.environ", {"LLAMA_SERVER_EXE": str(root / "missing.exe")}, clear=False):
                with patch("infrastructure.local_llm_parts.runtime_discovery.shutil.which", return_value=None):
                    with self.assertRaises(LocalLlmError) as ctx:
                        find_llama_server(root)
        self.assertEqual(ctx.exception.code, "local_llm_server_missing")

    def test_find_gguf_model_accepts_direct_and_named_models(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            direct = root / "direct.gguf"
            direct.write_bytes(b"gguf")
            nested = root / "models" / "llm" / "repo" / "model-Q4_K_M.gguf"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"gguf")

            self.assertEqual(find_gguf_model(root, str(direct)), direct.resolve())
            self.assertEqual(find_gguf_model(root, "model-Q4_K_M"), nested.resolve())
            with self.assertRaises(LocalLlmError):
                find_gguf_model(root, "")
            with self.assertRaises(LocalLlmError):
                find_gguf_model(root, "missing")

    def test_start_and_stop_local_server_process(self) -> None:
        process = Mock()
        process.poll.return_value = None
        server = Path("llama-server.exe")
        model = Path("model.gguf")

        with patch("infrastructure.local_llm_parts.runtime_server.subprocess.Popen", return_value=process) as popen:
            runtime_server._start_server_if_needed("127.0.0.1:1234:model", server, model, "alias", "127.0.0.1", 1234)

        self.assertIn("127.0.0.1:1234:model", runtime_server.SERVER_PROCESSES)
        self.assertIn("-m", popen.call_args.args[0])

        result = runtime_server.stop_local_llm(
            SimpleNamespace(id="local", base_url="http://127.0.0.1:1234/v1"),
            default_base_url="http://127.0.0.1:1234/v1",
        )

        self.assertEqual(result["killed"], 1)
        process.terminate.assert_called_once()

    def test_poll_until_ready_handles_success_exit_and_timeout(self) -> None:
        settings = AssistantExecutionSettings(command_tokens=[], path_hints=[], proxy="", timeout_s=10)
        with patch("infrastructure.local_llm_parts.runtime_server.request_json", return_value={"data": []}):
            message = runtime_server._poll_until_server_ready("http://127.0.0.1:1234/v1", settings, "missing")
        self.assertIn("Started local", message)

        process = Mock(returncode=2)
        process.poll.return_value = 2
        runtime_server.SERVER_PROCESSES["dead"] = process
        with patch(
            "infrastructure.local_llm_parts.runtime_server.request_json",
            side_effect=LocalLlmError("local_llm_unavailable", "refused", suggestion="retry"),
        ):
            with self.assertRaises(LocalLlmError) as ctx:
                runtime_server._poll_until_server_ready("http://127.0.0.1:1234/v1", settings, "dead")
        self.assertEqual(ctx.exception.code, "local_llm_server_exited")

        with (
            patch("infrastructure.local_llm_parts.runtime_server.time.time", side_effect=[0, 20]),
            patch("infrastructure.local_llm_parts.runtime_server.time.sleep"),
            patch(
                "infrastructure.local_llm_parts.runtime_server.request_json",
                side_effect=LocalLlmError("local_llm_unavailable", "refused", suggestion="retry"),
            ),
        ):
            with self.assertRaises(LocalLlmError) as timeout_ctx:
                runtime_server._poll_until_server_ready("http://127.0.0.1:1234/v1", settings, "missing")
        self.assertEqual(timeout_ctx.exception.code, "local_llm_start_timeout")

    def test_openai_local_runtime_reports_busy_local_port_before_autostart(self) -> None:
        settings = AssistantExecutionSettings(command_tokens=[], path_hints=[], proxy="", timeout_s=10)
        profile = SimpleNamespace(model="local-model")

        with (
            patch(
                "infrastructure.local_llm_parts.runtime_server.request_json",
                side_effect=LocalLlmError("local_llm_unavailable", "refused"),
            ),
            patch("infrastructure.local_llm_parts.runtime_server._localhost_port_accepts_connection", return_value=True),
            patch("infrastructure.local_llm_parts.runtime_server.find_gguf_model") as find_model,
        ):
            with self.assertRaises(LocalLlmError) as ctx:
                runtime_server.ensure_openai_local_runtime(profile, settings, "http://127.0.0.1:1234/v1")

        self.assertEqual(ctx.exception.code, "local_llm_port_in_use")
        self.assertIn("another base URL port", ctx.exception.suggestion)
        find_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
