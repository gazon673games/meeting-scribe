from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol


class TranscriptStore(Protocol):
    @property
    def current_human_log_path(self) -> Optional[Path]:
        ...

    @property
    def current_human_log_handle(self) -> Any:
        ...

    @property
    def realtime_transcript_path(self) -> Optional[Path]:
        ...

    def set_realtime_enabled(self, enabled: bool) -> None:
        ...

    def open_human_log(self) -> Optional[Path]:
        ...

    def close_human_log(self) -> None:
        ...

    def write_human_line(self, line: str) -> None:
        ...

    def close_realtime_transcript(self) -> None:
        ...

    def write_realtime_line(self, line: str) -> None:
        ...
