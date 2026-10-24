from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from application.local_paths import project_human_logs_dir, project_logs_dir

_FLUSH_INTERVAL_S = 1.0


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingSrtEntry:
    relative_ts: float
    label: str
    text: str


class FileTranscriptStore:
    def __init__(self, project_root: Path) -> None:
        self.project_root = Path(project_root)
        self._human_log_path: Path | None = None
        self._human_log_fh: TextIO | None = None
        self._realtime_enabled = False
        self._realtime_transcript_path: Path | None = None
        self._realtime_transcript_fh: TextIO | None = None
        self._srt_index: int = 0
        self._srt_pending: PendingSrtEntry | None = None
        self._srt_base_ts: float | None = None
        self._human_dirty = False
        self._human_last_flush = 0.0
        self._srt_dirty = False
        self._srt_last_flush = 0.0

    @property
    def current_human_log_path(self) -> Path | None:
        return self._human_log_path

    @property
    def current_human_log_handle(self) -> TextIO | None:
        return self._human_log_fh

    @property
    def realtime_transcript_path(self) -> Path | None:
        return self._realtime_transcript_path

    def set_realtime_enabled(self, enabled: bool) -> None:
        self._realtime_enabled = bool(enabled)
        if not self._realtime_enabled:
            self.close_realtime_transcript()

    def open_human_log(self) -> Path | None:
        self.close_human_log()
        try:
            directory = project_human_logs_dir(self.project_root, create=True)
            ts = time.strftime('%Y%m%d_%H%M%S')
            identity = uuid.uuid4().hex[:12]
            self._human_log_path = directory / f'chat_{ts}_{identity}.txt'
            self._human_log_fh = self._human_log_path.open(
                'a', encoding='utf-8'
            )
            self._human_log_fh.write(f'# Meeting Scribe chat log: {ts}\n')
            self._human_log_fh.flush()
            self._human_dirty = False
            self._human_last_flush = time.monotonic()
            return self._human_log_path
        except (OSError, ValueError):
            _logger.warning('Could not open the session log', exc_info=True)
            _flush_and_close(self._human_log_fh)
            self._human_log_fh = None
            self._human_log_path = None
            return None

    def close_human_log(self) -> None:
        _flush_and_close(self._human_log_fh)
        self._human_log_fh = None
        self._human_log_path = None
        self._human_dirty = False

    def write_human_line(self, line: str) -> None:
        if self._human_log_fh is None:
            return
        try:
            self._human_log_fh.write(str(line) + '\n')
            self._human_dirty = True
            self._flush_human_if_due()
        except (OSError, ValueError):
            _logger.debug(
                'Best-effort operation failed in write_human_line',
                exc_info=True,
            )

    def close_realtime_transcript(self) -> None:
        self._flush_srt_pending()
        _flush_and_close(self._realtime_transcript_fh)
        self._realtime_transcript_fh = None
        self._realtime_transcript_path = None
        self._srt_index = 0
        self._srt_pending = None
        self._srt_base_ts = None

    def write_realtime_srt_entry(
        self, ts: float, stream: str, text: str, *, speaker: str = ''
    ) -> None:
        if not self._realtime_enabled:
            return
        self._open_realtime_if_needed()
        if self._realtime_transcript_fh is None:
            return
        try:
            label = str(speaker or stream or 'mix')
            if self._srt_base_ts is None:
                self._srt_base_ts = ts
            relative_ts = max(0.0, ts - self._srt_base_ts)
            if self._srt_pending is not None:
                prev = self._srt_pending
                end_ts = max(prev.relative_ts + 0.5, relative_ts - 0.05)
                self._srt_index += 1
                self._realtime_transcript_fh.write(
                    f'{self._srt_index}\n'
                    f'{_srt_timestamp(prev.relative_ts)} --> {_srt_timestamp(end_ts)}\n'
                    f'{prev.label}: {prev.text}\n\n'
                )
                self._srt_dirty = True
                self._flush_realtime_if_due()
            self._srt_pending = PendingSrtEntry(
                relative_ts=relative_ts,
                label=label,
                text=str(text),
            )
        except (OSError, ValueError):
            _logger.debug(
                'Best-effort operation failed in write_realtime_srt_entry',
                exc_info=True,
            )

    def _flush_srt_pending(self) -> None:
        if self._srt_pending is None or self._realtime_transcript_fh is None:
            return
        try:
            prev = self._srt_pending
            end_ts = prev.relative_ts + 4.0
            self._srt_index += 1
            self._realtime_transcript_fh.write(
                f'{self._srt_index}\n'
                f'{_srt_timestamp(prev.relative_ts)} --> {_srt_timestamp(end_ts)}\n'
                f'{prev.label}: {prev.text}\n\n'
            )
            self._srt_dirty = True
            self._flush_realtime_if_due(force=True)
        except (OSError, ValueError):
            _logger.debug(
                'Best-effort operation failed in _flush_srt_pending',
                exc_info=True,
            )
        self._srt_pending = None

    def _open_realtime_if_needed(self) -> None:
        if not self._realtime_enabled:
            return
        if self._realtime_transcript_fh is not None:
            return
        try:
            directory = project_logs_dir(self.project_root, create=True)
            ts = time.strftime('%Y%m%d_%H%M%S')
            identity = uuid.uuid4().hex[:12]
            self._realtime_transcript_path = (
                directory / f'transcript_{ts}_{identity}.srt'
            )
            self._realtime_transcript_fh = self._realtime_transcript_path.open(
                'a', encoding='utf-8'
            )
            self._srt_index = 0
            self._srt_pending = None
            self._srt_base_ts = None
            self._srt_dirty = False
            self._srt_last_flush = time.monotonic()
        except (OSError, ValueError):
            _logger.warning(
                'Could not open the realtime transcript', exc_info=True
            )
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
        except (OSError, ValueError):
            _logger.debug(
                'Best-effort operation failed in _flush_human_if_due',
                exc_info=True,
            )

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
        except (OSError, ValueError):
            _logger.debug(
                'Best-effort operation failed in _flush_realtime_if_due',
                exc_info=True,
            )


def _flush_and_close(handle: TextIO | None) -> None:
    if handle is None:
        return
    try:
        handle.flush()
    except (OSError, ValueError):
        _logger.warning('Could not flush a transcript file', exc_info=True)
    finally:
        try:
            handle.close()
        except (OSError, ValueError):
            _logger.warning('Could not close a transcript file', exc_info=True)


def _srt_timestamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'
