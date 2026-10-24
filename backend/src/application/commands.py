from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from application.codex_config import CodexProfile


@dataclass(frozen=True)
class StartSessionCommand:
    source_count: int
    asr_enabled: bool
    model_name: str
    profile: str
    language: str


@dataclass(frozen=True)
class StopSessionCommand:
    run_offline_pass: bool = True
    wait: bool = False


@dataclass(frozen=True)
class SwitchProfileCommand:
    profile: str


@dataclass(frozen=True)
class InvokeAssistantCommand:
    profile: CodexProfile
    request_text: str
    source_label: str
    context_source: str
    context_label: str
    context_text: str | None = None
    human_log_path: Path | None = None
    human_log_fh: TextIO | None = None
    max_log_chars: int | None = None
    timeout_s: int | None = None
    fallback_max_log_chars: int | None = None
    fallback_timeout_s: int | None = None
