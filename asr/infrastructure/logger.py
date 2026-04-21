from __future__ import annotations

import json
import logging
import logging.handlers
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from asr.application.ports import AsrLoggerPort


class _EventFilter:
    """Drops high-frequency or disabled event types before they reach disk."""

    def __init__(self, *, write_segment_events: bool, audio_seen_min_interval_s: float) -> None:
        self._write_segments = write_segment_events
        self._audio_seen_min_s = float(audio_seen_min_interval_s)
        self._last_audio_seen: Dict[str, float] = {}

    def should_skip(self, rec: Dict[str, Any]) -> bool:
        typ = str(rec.get("type", ""))
        if typ == "segment" and not self._write_segments:
            return True
        if typ == "audio_seen":
            stream = str(rec.get("stream", ""))
            now = float(rec.get("ts", time.time()))
            if (now - self._last_audio_seen.get(stream, 0.0)) < self._audio_seen_min_s:
                return True
            self._last_audio_seen[stream] = now
        return False


@dataclass
class ASRLogger(AsrLoggerPort):
    root: Path
    session_id: str
    language: str = "ru"

    max_bytes: int = 25 * 1024 * 1024
    backup_count: int = 5
    audio_seen_min_interval_s: float = 5.0
    write_segment_events: bool = False

    def __post_init__(self) -> None:
        log_dir = self.root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"asr_{self.session_id}_{ts}.jsonl"

        self._filter = _EventFilter(
            write_segment_events=self.write_segment_events,
            audio_seen_min_interval_s=self.audio_seen_min_interval_s,
        )

        logger_name = f"asr.session.{self.session_id}.{ts}"
        self._log = logging.getLogger(logger_name)
        self._log.setLevel(logging.DEBUG)
        self._log.propagate = False

        handler = logging.handlers.RotatingFileHandler(
            self.path,
            mode="a",
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._log.addHandler(handler)
        self._handler = handler

    def write(self, rec: Dict[str, Any]) -> None:
        if not isinstance(rec, dict):
            return
        if self._filter.should_skip(rec):
            return
        try:
            self._log.info(json.dumps(rec, ensure_ascii=False))
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._handler.close()
            self._log.removeHandler(self._handler)
        except Exception:
            pass
