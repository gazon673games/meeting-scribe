from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from application.codex_assistant import CodexAssistantRequest, CodexAssistantResult, CodexExecutionSettings
from application.codex_config import CodexProfile
from application.local_paths import project_human_logs_dir
from application.codex_use_case import CodexRequestInput, CodexRequestUseCase
from transcription.infrastructure.file_transcript_context import FileTranscriptContextReader


class _CapturingAssistant:
    def __init__(self) -> None:
        self.last_prompt = ""

    def run(self, request: CodexAssistantRequest) -> CodexAssistantResult:
        self.last_prompt = request.prompt
        return CodexAssistantResult(ok=True, profile=request.profile.label, cmd=request.original_cmd, text="ok", dt_s=0.0)


class CodexContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_parent = Path(__file__).resolve().parents[1] / "tmp_tests"
        cls._tmp_parent.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp_parent, ignore_errors=True)

    def _project_root(self, name: str) -> Path:
        root = self._tmp_parent / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        return root

    def _request(self, root: Path, *, context_text=None, human_log_path=None) -> tuple[_CapturingAssistant, CodexRequestInput]:
        assistant = _CapturingAssistant()
        request = CodexRequestInput(
            user_text="ANSWER",
            profile=CodexProfile(id="fast", label="Fast", prompt="", reasoning_effort="low"),
            project_root=root,
            human_log_path=human_log_path,
            human_log_fh=None,
            max_log_chars=4000,
            answer_keyword="ANSWER",
            execution_settings=CodexExecutionSettings(
                command_tokens=["codex"],
                path_hints=[],
                proxy="",
                timeout_s=30,
            ),
            context_text=context_text,
        )
        return assistant, request

    def test_empty_current_transcript_does_not_fall_back_to_latest_human_log(self) -> None:
        root = self._project_root("empty_transcript")
        logs = project_human_logs_dir(root, create=True)
        (logs / "chat_old.txt").write_text("old interview question", encoding="utf-8")

        assistant, request = self._request(root, context_text="")
        CodexRequestUseCase(assistant, FileTranscriptContextReader()).execute(request)

        self.assertIn("(log is empty)", assistant.last_prompt)
        self.assertNotIn("old interview question", assistant.last_prompt)

    def test_current_transcript_is_used_when_provided(self) -> None:
        root = self._project_root("current_transcript")
        logs = project_human_logs_dir(root, create=True)
        (logs / "chat_old.txt").write_text("old interview question", encoding="utf-8")

        assistant, request = self._request(root, context_text="current transcript question")
        CodexRequestUseCase(assistant, FileTranscriptContextReader()).execute(request)

        self.assertIn("current transcript question", assistant.last_prompt)
        self.assertNotIn("old interview question", assistant.last_prompt)

    def test_latest_human_log_is_still_available_when_context_text_is_not_set(self) -> None:
        root = self._project_root("latest_human_log")
        logs = project_human_logs_dir(root, create=True)
        (logs / "chat_old.txt").write_text("old interview question", encoding="utf-8")

        assistant, request = self._request(root, context_text=None)
        CodexRequestUseCase(assistant, FileTranscriptContextReader()).execute(request)

        self.assertIn("old interview question", assistant.last_prompt)


if __name__ == "__main__":
    unittest.main()
