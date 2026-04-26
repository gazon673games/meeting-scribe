from __future__ import annotations

import json
import sys
from pathlib import Path

from settings.application.config_repository import ConfigRepository


def ensure_runtime_config(project_root: Path, repository: ConfigRepository) -> None:
    if not getattr(sys, "frozen", False):
        return

    bundled_root = Path(getattr(sys, "_MEIPASS", project_root))
    bundled_config = bundled_root / "config.json"
    if not bundled_config.exists():
        return

    if not repository.exists():
        try:
            repository.write(json.loads(bundled_config.read_text(encoding="utf-8")))
        except Exception:
            pass
        return

    try:
        current = repository.read()
        bundled = json.loads(bundled_config.read_text(encoding="utf-8"))
    except Exception:
        return

    if not isinstance(current, dict) or not isinstance(bundled, dict):
        return

    current_codex = current.get("codex", {})
    bundled_codex = bundled.get("codex", {})
    if not isinstance(current_codex, dict) or not isinstance(bundled_codex, dict):
        return

    has_profiles = bool(current_codex.get("profiles"))
    if "codex" in current and has_profiles:
        return

    current["codex"] = bundled_codex
    try:
        repository.write(current)
    except Exception:
        pass
