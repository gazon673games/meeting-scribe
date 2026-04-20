from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QTextCursor


class TranscriptMixin:
    def _init_transcript_state(self) -> None:
        # realtime transcript file (optional)
        self._rt_tr_to_file: bool = False
        self._rt_tr_path: Optional[Path] = None

        # human-readable session log (always on during ASR)
        self._human_log_path: Optional[Path] = None
        self._human_log_fh = None

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
        path = self.transcript_store.open_human_log()
        self._sync_transcript_store_refs()
        return path

    def _human_log_close(self) -> None:
        self.transcript_store.close_human_log()
        self._sync_transcript_store_refs()

    def _human_log_write_line(self, line: str) -> None:
        self.transcript_store.write_human_line(line)
        self._sync_transcript_store_refs()

    def _rt_open_if_needed(self) -> None:
        self.transcript_store.set_realtime_enabled(self._rt_tr_to_file)
        self._sync_transcript_store_refs()

    def _rt_close(self) -> None:
        self.transcript_store.set_realtime_enabled(False)
        self.transcript_store.close_realtime_transcript()
        self._sync_transcript_store_refs()

    def _rt_write_line(self, line: str) -> None:
        self._rt_open_if_needed()
        self.transcript_store.write_realtime_line(line)
        self._sync_transcript_store_refs()

    def _sync_transcript_store_refs(self) -> None:
        self._human_log_path = self.transcript_store.current_human_log_path
        self._human_log_fh = self.transcript_store.current_human_log_handle
        self._rt_tr_path = self.transcript_store.realtime_transcript_path
