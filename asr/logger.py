# --- File: D:\work\own\voice2textTest\asr\logger.py ---
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ASRLogger:
    root: Path
    session_id: str
    language: str = "ru"

    def __post_init__(self) -> None:
        self.dir = self.root / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"asr_{self.session_id}_{ts}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, rec: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
