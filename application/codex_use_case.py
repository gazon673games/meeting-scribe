from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from application.codex_assistant import (
    CodexAssistantPort,
    CodexAssistantRequest,
    CodexAssistantResult,
    CodexExecutionSettings,
)
from application.codex_config import CodexProfile
from application.codex_logs import read_human_log_tail
from application.codex_prompting import build_codex_prompt


@dataclass(frozen=True)
class CodexRequestInput:
    user_text: str
    profile: CodexProfile
    project_root: Path
    human_log_path: Optional[Path]
    human_log_fh: Any
    max_log_chars: int
    answer_keyword: str
    execution_settings: CodexExecutionSettings


class CodexRequestUseCase:
    def __init__(self, assistant: CodexAssistantPort) -> None:
        self._assistant = assistant

    def execute(self, request: CodexRequestInput) -> CodexAssistantResult:
        log_text = read_human_log_tail(
            project_root=Path(request.project_root),
            human_log_path=request.human_log_path,
            human_log_fh=request.human_log_fh,
            max_chars=int(request.max_log_chars),
        )
        prompt = build_codex_prompt(
            request.user_text,
            request.profile,
            log_text,
            answer_keyword=request.answer_keyword,
        )
        return self._assistant.run(
            CodexAssistantRequest(
                prompt=prompt,
                profile=request.profile,
                original_cmd=request.user_text,
                project_root=Path(request.project_root),
                settings=request.execution_settings,
            )
        )
