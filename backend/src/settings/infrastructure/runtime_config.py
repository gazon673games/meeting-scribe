from __future__ import annotations

import json
import sys
from pathlib import Path

from settings.application.config_repository import ConfigRepository


def _merge_bundled_codex_config(repository: ConfigRepository, current: dict, bundled: dict) -> None:
    current_codex = current.get("codex", {})
    bundled_codex = bundled.get("codex", {})
    if not isinstance(current_codex, dict) or not isinstance(bundled_codex, dict):
        return
    if "codex" in current and bool(current_codex.get("profiles")):
        return
    current["codex"] = bundled_codex
    try:
        repository.write(current)
    except Exception:
        pass


def ensure_runtime_config(project_root: Path, repository: ConfigRepository) -> None:
    if not getattr(sys, "frozen", False):
        return

    bundled_config = Path(getattr(sys, "_MEIPASS", project_root)) / "config.json"
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

    _merge_bundled_codex_config(repository, current, bundled)
