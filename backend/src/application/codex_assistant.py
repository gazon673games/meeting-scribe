from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol

from application.codex_config import CodexProfile


@dataclass(frozen=True)
class CodexExecutionSettings:
    command_tokens: List[str]
    path_hints: List[str]
    proxy: str
    timeout_s: int


@dataclass(frozen=True)
class CodexAssistantRequest:
    prompt: str
    profile: CodexProfile
    original_cmd: str
    project_root: Path
    settings: CodexExecutionSettings


@dataclass(frozen=True)
class CodexAssistantResult:
    ok: bool
    profile: str
    cmd: str
    text: str
    dt_s: float


class CodexAssistantPort(Protocol):
    def run(self, request: CodexAssistantRequest) -> CodexAssistantResult:
        ...
