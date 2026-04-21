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
from application.codex_prompting import build_codex_prompt
from transcription.application.transcript_context import TranscriptContextReader, trim_text_tail


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
    context_text: Optional[str] = None


class CodexRequestUseCase:
    def __init__(self, assistant: CodexAssistantPort, context_reader: TranscriptContextReader) -> None:
        self._assistant = assistant
        self._context_reader = context_reader

    def execute(self, request: CodexRequestInput) -> CodexAssistantResult:
        if request.context_text is not None:
            log_text = trim_text_tail(request.context_text, max_chars=int(request.max_log_chars))
        else:
            log_text = self._context_reader.read_human_log_tail(
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
