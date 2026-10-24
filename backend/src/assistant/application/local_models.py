from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from application.codex_config import CodexProfile

ModelStatusSink = Callable[[dict[str, Any]], None]


class LocalModelRuntimePort(Protocol):
    def start(
        self, profile: CodexProfile, project_root: Path, emit: ModelStatusSink
    ) -> dict[str, Any]: ...

    def stop(self, profile: CodexProfile) -> dict[str, Any]: ...
