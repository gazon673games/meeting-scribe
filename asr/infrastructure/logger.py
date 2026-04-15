from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from asr.application.ports import AsrLoggerPort


@dataclass
class ASRLogger(AsrLoggerPort):
    root: Path
    session_id: str
    language: str = "ru"

    # rotation
    max_bytes: int = 25 * 1024 * 1024  # 25MB
    backup_count: int = 5

    # Step 4: buffered IO
    flush_interval_s: float = 0.8          # flush at most ~1x/sec
    flush_every_n: int = 64                # or every N records (whichever first)

    # Step 4: reduce noisy events
    audio_seen_min_interval_s: float = 5.0  # per stream
    write_segment_events: bool = False      # disable "segment" by default (normal mode)

    def __post_init__(self) -> None:
        self.dir = self.root / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.dir / f"asr_{self.session_id}_{ts}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

        try:
            self._bytes_written = int(self.path.stat().st_size)
        except Exception:
            self._bytes_written = 0

        # buffering state
        self._buf: list[str] = []
        self._buf_bytes: int = 0
        self._since_flush: int = 0
        self._last_flush_ts: float = time.time()

        # throttles
        self._last_audio_seen_ts_by_stream: Dict[str, float] = {}

    def _should_rotate(self, extra_bytes: int = 0) -> bool:
        if self.max_bytes <= 0:
            return False
        try:
            return int(self._bytes_written + int(extra_bytes)) >= int(self.max_bytes)
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
                        # Windows: fallback to copy+remove
                        try:
                            import shutil

                            shutil.copy2(src, dst)
                            try:
                                src.unlink()
                            except Exception:
                                pass
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

    def _maybe_skip(self, rec: Dict[str, Any]) -> bool:
        typ = str(rec.get("type", ""))
        if typ == "segment" and not bool(self.write_segment_events):
            return True

        if typ == "audio_seen":
            stream = str(rec.get("stream", ""))
            now = float(rec.get("ts", time.time()))
            last = float(self._last_audio_seen_ts_by_stream.get(stream, 0.0))
            if (now - last) < float(self.audio_seen_min_interval_s):
                return True
            self._last_audio_seen_ts_by_stream[stream] = now

        return False

    def _flush_buffer(self) -> None:
        if not self._buf:
            self._last_flush_ts = time.time()
            self._since_flush = 0
            self._buf_bytes = 0
            return

        data = "".join(self._buf)
        try:
            self._fh.write(data)
            self._fh.flush()
        except Exception:
            # best-effort: drop buffer if disk is unhappy
            pass

        try:
            self._bytes_written += int(self._buf_bytes)
        except Exception:
            self._bytes_written += len(data.encode("utf-8", errors="ignore"))

        self._buf.clear()
        self._buf_bytes = 0
        self._since_flush = 0
        self._last_flush_ts = time.time()

    def write(self, rec: Dict[str, Any]) -> None:
        if not isinstance(rec, dict):
            return
        try:
            if self._maybe_skip(rec):
                return
        except Exception:
            pass

        try:
            s = json.dumps(rec, ensure_ascii=False) + "\n"
        except Exception:
            return

        # rotate decision based on buffered + this record
        extra_bytes = 0
        try:
            extra_bytes = len(s.encode("utf-8", errors="ignore"))
        except Exception:
            extra_bytes = len(s)

        # if adding this record would exceed limit, flush buffer first, then rotate if needed
        try:
            if self._should_rotate(extra_bytes=self._buf_bytes + extra_bytes):
                self._flush_buffer()
                if self._should_rotate(extra_bytes=extra_bytes):
                    self._rotate()
        except Exception:
            pass

        self._buf.append(s)
        self._buf_bytes += int(extra_bytes)
        self._since_flush += 1

        now = time.time()
        if self._since_flush >= int(self.flush_every_n) or (now - float(self._last_flush_ts)) >= float(self.flush_interval_s):
            self._flush_buffer()

    def close(self) -> None:
        try:
            self._flush_buffer()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
