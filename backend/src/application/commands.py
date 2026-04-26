from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
    context_text: Optional[str] = None
    human_log_path: Optional[Path] = None
    human_log_fh: Any = None
    max_log_chars: Optional[int] = None
    timeout_s: Optional[int] = None
    fallback_max_log_chars: Optional[int] = None
    fallback_timeout_s: Optional[int] = None
