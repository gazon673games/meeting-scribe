from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from application.local_paths import project_human_logs_dir, project_logs_dir


class FileTranscriptStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self._human_log_path: Optional[Path] = None
        self._human_log_fh: Any = None
        self._realtime_enabled = False
        self._realtime_transcript_path: Optional[Path] = None
        self._realtime_transcript_fh: Any = None

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

    def write_human_line(self, line: str) -> None:
        if self._human_log_fh is None:
            return
        try:
            self._human_log_fh.write(str(line) + "\n")
            self._human_log_fh.flush()
        except Exception:
            pass

    def close_realtime_transcript(self) -> None:
        try:
            if self._realtime_transcript_fh is not None:
                self._realtime_transcript_fh.flush()
                self._realtime_transcript_fh.close()
        except Exception:
            pass
        self._realtime_transcript_fh = None
        self._realtime_transcript_path = None

    def write_realtime_line(self, line: str) -> None:
        if not self._realtime_enabled:
            return
        self._open_realtime_if_needed()
        if self._realtime_transcript_fh is None:
            return
        try:
            self._realtime_transcript_fh.write(str(line) + "\n")
        except Exception:
            pass

    def _open_realtime_if_needed(self) -> None:
        if not self._realtime_enabled:
            return
        if self._realtime_transcript_fh is not None:
            return
        try:
            directory = project_logs_dir(self.project_root, create=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._realtime_transcript_path = directory / f"transcript_{ts}.txt"
            self._realtime_transcript_fh = self._realtime_transcript_path.open("a", encoding="utf-8")
        except Exception:
            self._realtime_transcript_fh = None
            self._realtime_transcript_path = None
