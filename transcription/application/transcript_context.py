from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol


class TranscriptContextReader(Protocol):
    def read_human_log_tail(
        self,
        *,
        project_root: Path,
        human_log_path: Optional[Path],
        human_log_fh: Any,
        max_chars: int,
    ) -> str:
        ...


def trim_text_tail(text: str, *, max_chars: int) -> str:
    max_chars = max(2000, int(max_chars))
    out = str(text or "")
    if len(out) > max_chars:
        out = "[context tail]\n" + out[-max_chars:]
    return out.strip()
