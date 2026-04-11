from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QTextCursor


class TranscriptMixin:
    def _clear_transcript(self) -> None:
        self.txt_tr.clear()

    @staticmethod
    def _fmt_ts(ts: float) -> str:
        try:
            local_time = time.localtime(ts)
            return time.strftime("%H:%M:%S", local_time)
        except Exception:
            return "??:??:??"

    def _is_at_bottom(self, margin_px: int = 6) -> bool:
        scrollbar = self.txt_tr.verticalScrollBar()
        return int(scrollbar.value()) >= int(scrollbar.maximum()) - int(margin_px)

    def _append_transcript_line(self, line: str) -> None:
        max_chars = 400_000
        if self.txt_tr.document().characterCount() > max_chars:
            self.txt_tr.clear()
            self.txt_tr.append("[transcript cleared: too large]")

        stick = self._is_at_bottom()
        self.txt_tr.append(line)

        if stick:
            self.txt_tr.moveCursor(QTextCursor.End)
            self.txt_tr.ensureCursorVisible()

        self._human_log_write_line(line)

        if self._rt_tr_to_file:
            self._rt_write_line(line)

    def _warn_throttle(self, msg: str, *, min_interval_s: float = 1.2) -> None:
        now = time.time()
        if (now - float(self._last_warn_ts)) < float(min_interval_s):
            return
        self._last_warn_ts = now
        self._append_transcript_line(f"[{self._fmt_ts(now)}] WARNING: {msg}")

    def _human_log_open_session(self) -> Optional[Path]:
        self._human_log_close()
        try:
            directory = self.project_root / "human_logs"
            directory.mkdir(parents=True, exist_ok=True)
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

    def _human_log_close(self) -> None:
        try:
            if self._human_log_fh is not None:
                self._human_log_fh.flush()
                self._human_log_fh.close()
        except Exception:
            pass
        self._human_log_fh = None
        self._human_log_path = None

    def _human_log_write_line(self, line: str) -> None:
        if self._human_log_fh is None:
            return
        try:
            self._human_log_fh.write(line + "\n")
            self._human_log_fh.flush()
        except Exception:
            pass

    def _rt_open_if_needed(self) -> None:
        if not self._rt_tr_to_file:
            return
        if self._rt_tr_fh is not None:
            return
        try:
            directory = self.project_root / "logs"
            directory.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._rt_tr_path = directory / f"transcript_{ts}.txt"
            self._rt_tr_fh = self._rt_tr_path.open("a", encoding="utf-8")
        except Exception:
            self._rt_tr_fh = None
            self._rt_tr_path = None

    def _rt_close(self) -> None:
        try:
            if self._rt_tr_fh is not None:
                self._rt_tr_fh.flush()
                self._rt_tr_fh.close()
        except Exception:
            pass
        self._rt_tr_fh = None
        self._rt_tr_path = None

    def _rt_write_line(self, line: str) -> None:
        self._rt_open_if_needed()
        if self._rt_tr_fh is None:
            return
        try:
            self._rt_tr_fh.write(line + "\n")
        except Exception:
            pass
