from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from application.local_paths import project_human_logs_dir, project_logs_dir

_FLUSH_INTERVAL_S = 1.0


class FileTranscriptStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self._human_log_path: Optional[Path] = None
        self._human_log_fh: Any = None
        self._realtime_enabled = False
        self._realtime_transcript_path: Optional[Path] = None
        self._realtime_transcript_fh: Any = None
        self._srt_index: int = 0
        self._srt_pending: Optional[Dict[str, Any]] = None
        self._srt_base_ts: Optional[float] = None
        self._human_dirty = False
        self._human_last_flush = 0.0
        self._srt_dirty = False
        self._srt_last_flush = 0.0

    @property
    def current_human_log_path(self) -> Optional[Path]:
        return self._human_log_path

    @property
    def current_human_log_handle(self) -> Any:
        return self._human_log_fh

    @property
    def realtime_transcript_path(self) -> Optional[Path]:
        return self._realtime_transcript_path

    def set_realtime_enabled(self, enabled: bool) -> None:
        self._realtime_enabled = bool(enabled)
        if not self._realtime_enabled:
            self.close_realtime_transcript()

    def open_human_log(self) -> Optional[Path]:
        self.close_human_log()
        try:
            directory = project_human_logs_dir(self.project_root, create=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._human_log_path = directory / f"chat_{ts}.txt"
            self._human_log_fh = self._human_log_path.open("a", encoding="utf-8")
            self._human_log_fh.write(f"# Meeting Scribe chat log: {ts}\n")
            self._human_log_fh.flush()
            self._human_dirty = False
            self._human_last_flush = time.monotonic()
            return self._human_log_path
        except Exception:
            self._human_log_fh = None
            self._human_log_path = None
            return None

    def close_human_log(self) -> None:
        try:
            if self._human_log_fh is not None:
                self._human_log_fh.flush()
                self._human_log_fh.close()
        except Exception:
            pass
        self._human_log_fh = None
        self._human_log_path = None
        self._human_dirty = False

    def write_human_line(self, line: str) -> None:
        if self._human_log_fh is None:
            return
        try:
            self._human_log_fh.write(str(line) + "\n")
            self._human_dirty = True
            self._flush_human_if_due()
        except Exception:
            pass

    def close_realtime_transcript(self) -> None:
        self._flush_srt_pending()
        try:
            if self._realtime_transcript_fh is not None:
                self._realtime_transcript_fh.flush()
                self._realtime_transcript_fh.close()
        except Exception:
            pass
        self._realtime_transcript_fh = None
        self._realtime_transcript_path = None
        self._srt_index = 0
        self._srt_pending = None
        self._srt_base_ts = None

    def write_realtime_srt_entry(self, ts: float, stream: str, text: str, *, speaker: str = "") -> None:
        if not self._realtime_enabled:
            return
        self._open_realtime_if_needed()
        if self._realtime_transcript_fh is None:
            return
        try:
            label = str(speaker or stream or "mix")
            if self._srt_base_ts is None:
                self._srt_base_ts = ts
            relative_ts = max(0.0, ts - self._srt_base_ts)
            if self._srt_pending is not None:
                prev = self._srt_pending
                end_ts = max(prev["relative_ts"] + 0.5, relative_ts - 0.05)
                self._srt_index += 1
                self._realtime_transcript_fh.write(
                    f"{self._srt_index}\n"
                    f"{_srt_timestamp(prev['relative_ts'])} --> {_srt_timestamp(end_ts)}\n"
                    f"{prev['label']}: {prev['text']}\n\n"
                )
                self._srt_dirty = True
                self._flush_realtime_if_due()
            self._srt_pending = {"relative_ts": relative_ts, "label": label, "text": str(text)}
        except Exception:
            pass

    def _flush_srt_pending(self) -> None:
        if self._srt_pending is None or self._realtime_transcript_fh is None:
            return
        try:
            prev = self._srt_pending
            end_ts = prev["relative_ts"] + 4.0
            self._srt_index += 1
            self._realtime_transcript_fh.write(
                f"{self._srt_index}\n"
                f"{_srt_timestamp(prev['relative_ts'])} --> {_srt_timestamp(end_ts)}\n"
                f"{prev['label']}: {prev['text']}\n\n"
            )
            self._srt_dirty = True
            self._flush_realtime_if_due(force=True)
        except Exception:
            pass
        self._srt_pending = None

    def _open_realtime_if_needed(self) -> None:
        if not self._realtime_enabled:
            return
        if self._realtime_transcript_fh is not None:
            return
        try:
            directory = project_logs_dir(self.project_root, create=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._realtime_transcript_path = directory / f"transcript_{ts}.srt"
            self._realtime_transcript_fh = self._realtime_transcript_path.open("a", encoding="utf-8")
            self._srt_index = 0
            self._srt_pending = None
            self._srt_base_ts = None
            self._srt_dirty = False
            self._srt_last_flush = time.monotonic()
        except Exception:
            self._realtime_transcript_fh = None
            self._realtime_transcript_path = None

    def _flush_human_if_due(self, *, force: bool = False) -> None:
        if self._human_log_fh is None or not self._human_dirty:
            return
        now = time.monotonic()
        if not force and now - self._human_last_flush < _FLUSH_INTERVAL_S:
            return
        try:
            self._human_log_fh.flush()
            self._human_dirty = False
            self._human_last_flush = now
        except Exception:
            pass

    def _flush_realtime_if_due(self, *, force: bool = False) -> None:
        if self._realtime_transcript_fh is None or not self._srt_dirty:
            return
        now = time.monotonic()
        if not force and now - self._srt_last_flush < _FLUSH_INTERVAL_S:
            return
        try:
            self._realtime_transcript_fh.flush()
            self._srt_dirty = False
            self._srt_last_flush = now
        except Exception:
            pass


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
