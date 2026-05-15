from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from application.codex_config import CodexProfile
from assistant.application.provider import AssistantExecutionSettings
from infrastructure.codex_cli_parts.commands import (
    build_exec_cmd,
    interactive_login_cmd,
    new_console_creationflags,
    process_output_text,
    read_output_file,
)
from infrastructure.codex_cli_parts.resolver import CodexCommandResolver


class CodexCliPartsTests(unittest.TestCase):
    def test_build_exec_cmd_includes_profile_model_effort_and_extra_args(self) -> None:
        profile = CodexProfile(
            id="deep",
            label="Deep",
            prompt="",
            model="gpt-5.2",
            reasoning_effort="high",
            codex_profile="work",
            extra_args=["--sandbox", "read-only", ""],
        )

        cmd = build_exec_cmd(["codex"], profile, Path("out.txt"))

        self.assertEqual(cmd[:4], ["codex", "exec", "--color", "never"])
        self.assertIn("-m", cmd)
        self.assertIn("gpt-5.2", cmd)
        self.assertIn('model_reasoning_effort="high"', cmd)
        self.assertIn("work", cmd)
        self.assertEqual(cmd[-2:], ["out.txt", "-"])

    def test_output_reading_prefers_saved_file_then_stdout_then_stderr(self) -> None:
        proc = subprocess.CompletedProcess(["codex"], 0, stdout="stdout\n", stderr="stderr\n")
        with tempfile.TemporaryDirectory() as raw_root:
            out_path = Path(raw_root) / "answer.txt"

            self.assertEqual(read_output_file(out_path, proc), "stdout")
            out_path.write_text(" file answer \n", encoding="utf-8")
            self.assertEqual(read_output_file(out_path, proc), "file answer")

        self.assertEqual(process_output_text(proc), "stdout")
        self.assertEqual(process_output_text(subprocess.CompletedProcess(["codex"], 1, stdout="", stderr="bad\n")), "bad")

    def test_login_command_keeps_windows_console_open(self) -> None:
        with (
            patch("infrastructure.codex_cli_parts.commands.os.name", "nt"),
            patch.dict("os.environ", {"COMSPEC": "cmd.exe"}),
        ):
            self.assertEqual(
                interactive_login_cmd(["cmd.exe", "/d", "/c", "codex"], ["login"]),
                ["cmd.exe", "/d", "/k", "codex", "login"],
            )
            self.assertEqual(
                interactive_login_cmd(["codex"], ["login"]),
                ["cmd.exe", "/d", "/k", "codex", "login"],
            )
            with patch("infrastructure.codex_cli_parts.commands.subprocess.CREATE_NEW_CONSOLE", 16, create=True):
                self.assertEqual(new_console_creationflags(), 16)

        with patch("infrastructure.codex_cli_parts.commands.os.name", "posix"):
            self.assertEqual(interactive_login_cmd(["codex"], ["login"]), ["codex", "login"])
            self.assertEqual(new_console_creationflags(), 0)

    def test_resolver_uses_direct_path_hints_path_and_where_fallback(self) -> None:
        settings = AssistantExecutionSettings(command_tokens=["codex", "--flag"], path_hints=[], proxy="", timeout_s=10)
        resolver = CodexCommandResolver()
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            direct = root / "codex.cmd"
            direct.write_text("@echo off", encoding="utf-8")
            direct_settings = AssistantExecutionSettings(
                command_tokens=[str(direct), "--flag"],
                path_hints=[],
                proxy="",
                timeout_s=10,
            )

            cmd, source = resolver.resolve(direct_settings)  # type: ignore[misc]
            self.assertEqual(source, "direct_path")
            self.assertEqual(cmd[-2:], [str(direct), "--flag"])

            hint_dir = root / "hint"
            hint_dir.mkdir()
            hinted = hint_dir / "codex.exe"
            hinted.write_text("exe", encoding="utf-8")
            hint_settings = AssistantExecutionSettings(
                command_tokens=["codex"],
                path_hints=[str(hint_dir)],
                proxy="",
                timeout_s=10,
            )
            with patch("infrastructure.codex_cli_parts.resolver.shutil.which", return_value=None):
                cmd, source = resolver.resolve(hint_settings)  # type: ignore[misc]
            self.assertEqual(source, f"hint:{hint_dir}")
            self.assertEqual(cmd, [str(hinted)])

        with patch("infrastructure.codex_cli_parts.resolver.shutil.which", return_value="C:/bin/codex.exe"):
            cmd, source = resolver.resolve(settings)  # type: ignore[misc]
        self.assertEqual(source, "path:codex")
        self.assertEqual(cmd, ["C:/bin/codex.exe", "--flag"])

        where_result = Mock(returncode=0, stdout="C:\\tools\\codex.exe\n")

        def exists_for_where(path: Path) -> bool:
            return str(path) == "C:\\tools\\codex.exe"

        with (
            patch("infrastructure.codex_cli_parts.resolver.shutil.which", return_value=None),
            patch("infrastructure.codex_cli_parts.resolver.Path.exists", exists_for_where),
            patch("infrastructure.codex_cli_parts.resolver.subprocess.run", return_value=where_result),
        ):
            cmd, source = resolver.resolve(settings)  # type: ignore[misc]
        self.assertEqual(source, "where")
        self.assertEqual(cmd[-1], "--flag")


if __name__ == "__main__":
    unittest.main()
