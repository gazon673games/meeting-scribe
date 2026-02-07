# --- File: D:\work\own\voice2textTest\asr\logger.py ---
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ASRLogger:
    root: Path
    session_id: str
    language: str = "ru"

    # NEW: rotation
    max_bytes: int = 25 * 1024 * 1024  # 25MB
    backup_count: int = 5

    def __post_init__(self) -> None:
        self.dir = self.root / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"asr_{self.session_id}_{ts}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")
        self._bytes_written = 0

        try:
            self._bytes_written = self.path.stat().st_size
        except Exception:
            self._bytes_written = 0

    def _should_rotate(self) -> bool:
        if self.max_bytes <= 0:
            return False
        try:
            return int(self._bytes_written) >= int(self.max_bytes)
        except Exception:
            return False

    def _rotate(self) -> None:
        # close current
        try:
            self._fh.close()
        except Exception:
            pass

        # shift older backups: .(backup_count-1) -> .backup_count
        try:
            for i in range(int(self.backup_count), 0, -1):
                src = Path(str(self.path) + ("" if i == 1 else f".{i-1}"))
                dst = Path(str(self.path) + f".{i}")
                if src.exists():
                    if dst.exists():
                        try:
                            dst.unlink()
                        except Exception:
                            pass
                    try:
                        src.rename(dst)
                    except Exception:
                        # If rename fails (Windows lock/race), try copy+remove
                        try:
                            import shutil

                            shutil.copy2(src, dst)
                            if i != 1:
                                src.unlink()
                            else:
                                src.unlink()
                        except Exception:
                            pass

            # rename base to .1 (if exists and not already moved)
            if self.path.exists():
                b1 = Path(str(self.path) + ".1")
                if not b1.exists():
                    try:
                        self.path.rename(b1)
                    except Exception:
                        pass
        except Exception:
            pass

        # reopen fresh
        self._fh = self.path.open("a", encoding="utf-8")
        self._bytes_written = 0

    def write(self, rec: Dict[str, Any]) -> None:
        s = json.dumps(rec, ensure_ascii=False) + "\n"
        try:
            if self._should_rotate():
                self._rotate()
        except Exception:
            pass

        self._fh.write(s)
        self._fh.flush()

        try:
            self._bytes_written += len(s.encode("utf-8", errors="ignore"))
        except Exception:
            self._bytes_written += len(s)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
