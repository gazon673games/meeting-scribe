from __future__ import annotations

from pathlib import Path

from application.codex_assistant import CodexExecutionSettings


def settings_project_root(settings: CodexExecutionSettings) -> Path:
    raw = getattr(settings, "project_root", None)
    return Path(raw).resolve() if raw else Path.cwd().resolve()


def codex_home(project_root: Path) -> Path:
    return Path(project_root).resolve() / ".local" / "codex_home"
