from __future__ import annotations

import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

from assistant.application.provider import AssistantExecutionSettings
from assistant.application.provider import AssistantProviderError, AssistantProviderRequest
from application.codex_config import CodexProfile
from infrastructure.codex_cli import CodexCliRunner
from infrastructure.codex_cli_parts.errors import codex_not_found_error, provider_info_from_error


def _canonical_path(value: object) -> Path:
    return Path(value).resolve()


class _Resolver:
    def resolve(self, settings):  # noqa: ANN001
        return ["codex"], "test"


class _Response:
    status = 200

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Opener:
    def __init__(self, result=None, error=None) -> None:  # noqa: ANN001
        self.result = result or _Response()
        self.error = error
        self.request = None
        self.timeout = None

    def open(self, request, timeout=None):  # noqa: ANN001
        self.request = request
        self.timeout = timeout
        if self.error is not None:
            raise self.error
        return self.result


def _settings(root: Path) -> AssistantExecutionSettings:
    return AssistantExecutionSettings(
        command_tokens=["codex"],
        path_hints=[],
        proxy="",
        timeout_s=30,
        project_root=root,
    )


class CodexCliRunnerTests(unittest.TestCase):
    def test_status_reports_auth_error_for_local_codex_home(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            runner = CodexCliRunner()
            runner._resolver = _Resolver()  # type: ignore[assignment]
            completed = subprocess.CompletedProcess(["codex", "login", "status"], 1, stdout="Not logged in\n", stderr="")

            with patch("infrastructure.codex_cli.subprocess.run", return_value=completed) as run:
                status = runner.status(_settings(root))

            self.assertFalse(status.available)
            self.assertEqual(status.error_code, "auth_error")
            self.assertTrue(status.auth_required)
            self.assertTrue(status.login_supported)
            self.assertIn("Not logged in", status.message)
            call = run.call_args
            self.assertEqual(call.args[0], ["codex", "login", "status"])
            self.assertEqual(_canonical_path(call.kwargs["env"]["CODEX_HOME"]), _canonical_path(root / ".local" / "codex_home"))
            self.assertEqual(_canonical_path(call.kwargs["cwd"]), _canonical_path(root))

    def test_status_reports_logged_in_without_model_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            runner = CodexCliRunner()
            runner._resolver = _Resolver()  # type: ignore[assignment]
            completed = subprocess.CompletedProcess(
                ["codex", "login", "status"],
                0,
                stdout="Logged in using ChatGPT\n",
                stderr="",
            )

            with patch("infrastructure.codex_cli.subprocess.run", return_value=completed) as run:
                status = runner.status(_settings(root))

            cmd = run.call_args.args[0]
            self.assertTrue(status.available)
            self.assertIn("Logged in using ChatGPT", status.message)
            self.assertNotIn("exec", cmd)

    def test_start_login_launches_local_codex_login(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            runner = CodexCliRunner()
            runner._resolver = _Resolver()  # type: ignore[assignment]

            with patch("infrastructure.codex_cli.subprocess.Popen", return_value=Mock()) as popen:
                result = runner.start_login(_settings(root))

            cmd = [str(part) for part in popen.call_args.args[0]]
            self.assertTrue(result.started)
            self.assertIn("login", cmd)
            self.assertNotIn("exec", cmd)
            self.assertEqual(_canonical_path(popen.call_args.kwargs["env"]["CODEX_HOME"]), _canonical_path(root / ".local" / "codex_home"))
            self.assertEqual(_canonical_path(popen.call_args.kwargs["cwd"]), _canonical_path(root))

    def test_ping_reports_api_reachable_without_codex_exec(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            runner = CodexCliRunner()
            opener = _Opener(error=urllib.error.HTTPError("https://api.openai.com/v1/models", 401, "Unauthorized", {}, None))

            with patch("infrastructure.codex_cli.urllib.request.build_opener", return_value=opener):
                result = runner.ping(_settings(root))

            self.assertTrue(result.ok)
            self.assertEqual(result.status_code, 401)
            self.assertIn("reachable", result.message)
            self.assertEqual(opener.request.full_url, "https://api.openai.com/v1/models")

    def test_ping_reports_network_error(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            runner = CodexCliRunner()
            opener = _Opener(error=urllib.error.URLError("offline"))

            with patch("infrastructure.codex_cli.urllib.request.build_opener", return_value=opener):
                result = runner.ping(_settings(root))

        self.assertFalse(result.ok)
        self.assertEqual(result.error_code, "network_error")
        self.assertTrue(result.retryable)

    def test_run_executes_codex_command_and_private_result_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            profile = CodexProfile(id="default", label="Default", prompt="", model="gpt-5.4-mini")
            runner = CodexCliRunner()
            runner._resolver = _Resolver()  # type: ignore[assignment]
            request = AssistantProviderRequest(
                prompt="hello",
                profile=profile,
                original_cmd="ANSWER",
                project_root=root,
                settings=_settings(root),
            )
            completed = subprocess.CompletedProcess(["codex"], 0, stdout="", stderr="")

            with (
                patch("infrastructure.codex_cli.subprocess.run", return_value=completed) as run,
                patch("infrastructure.codex_cli.read_output_file", return_value="assistant answer"),
            ):
                result = runner.run(request)

        self.assertTrue(result.ok)
        self.assertEqual(result.text, "assistant answer")
        self.assertEqual(run.call_args.kwargs["input"], "hello")
        self.assertEqual(runner._result(True, profile, "cmd", "ok", 0).provider, "codex")
        self.assertEqual(
            runner._error_result(profile, "cmd", AssistantProviderError("boom", "failed"), 0).error_code,
            "boom",
        )
        self.assertFalse(runner._info_from_error(codex_not_found_error()).available)
        self.assertFalse(provider_info_from_error(codex_not_found_error(runtime=True), provider_id="codex", provider_label="Codex").login_supported)


if __name__ == "__main__":
    unittest.main()
