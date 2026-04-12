from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol

from audio.types import AudioFormat


class WavRecorder(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def start_recording(self, path: Path, fmt: AudioFormat) -> None:
        ...

    def stop_recording(self) -> None:
        ...

    def is_recording(self) -> bool:
        ...

    def target_path(self) -> Optional[Path]:
        ...

    def last_error(self) -> Optional[str]:
        ...


class WavRecorderFactory(Protocol):
    def available(self) -> bool:
        ...

    def create(self, output_queue: Any) -> WavRecorder:
        ...
