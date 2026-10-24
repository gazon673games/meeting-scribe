from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class OfflineAsrRequest:
    project_root: Path
    wav_path: Path
    out_txt: Path
    model_name: str
    language: str | None
    models_dir: Path | None = None


class OfflineAsrRunnerPort(Protocol):
    def available(self) -> bool: ...

    def run(self, request: OfflineAsrRequest) -> Path: ...
