from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def read_human_log_tail(
    *,
    project_root: Path,
    human_log_path: Optional[Path],
    human_log_fh: Any,
    max_chars: int,
) -> str:
    path = _resolve_human_log_path(project_root=project_root, human_log_path=human_log_path)
    if path is None or not path.exists():
        return ""

    try:
        if human_log_fh is not None:
            human_log_fh.flush()
    except Exception:
        pass

    max_chars = max(2000, int(max_chars))
    max_bytes = max_chars * 4 + 4096

    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            start = max(0, int(size) - int(max_bytes))
            fh.seek(start, os.SEEK_SET)
            raw = fh.read()
    except Exception:
        return ""

    text = raw.decode("utf-8", errors="ignore")
    if len(text) > max_chars:
        text = text[-max_chars:]
        text = "[log tail]\n" + text
    return text.strip()


def _resolve_human_log_path(*, project_root: Path, human_log_path: Optional[Path]) -> Optional[Path]:
    if human_log_path is not None and Path(human_log_path).exists():
        return Path(human_log_path)

    logs_dir = project_root / "human_logs"
    if not logs_dir.exists():
        return None

    try:
        files = [x for x in logs_dir.glob("chat_*.txt") if x.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        return None
    return files[0] if files else None
