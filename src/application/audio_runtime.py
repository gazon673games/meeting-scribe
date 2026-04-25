from __future__ import annotations

from typing import Any, List, Optional, Protocol

from audio.domain.formats import AudioFormat
from audio.domain.ports import AudioFilter, AudioSource
from audio.domain.types import TapMode


class AudioRuntimePort(Protocol):
    @property
    def format(self) -> AudioFormat:
        ...

    def is_running(self) -> bool:
        ...

    def set_tap_queue(self, tap_queue: Optional[Any]) -> None:
        ...

    def set_tap_config(
        self,
        *,
        mode: TapMode = "both",
        sources: Optional[List[str]] = None,
        drop_threshold: float = 0.85,
    ) -> None:
        ...

    def add_source(self, src: AudioSource) -> None:
        ...

    def remove_source(self, name: str) -> None:
        ...

    def add_master_filter(self, flt: AudioFilter) -> None:
        ...

    def set_source_enabled(self, name: str, enabled: bool) -> None:
        ...

    def set_source_delay_ms(self, name: str, delay_ms: float) -> None:
        ...

    def enable_auto_sync(self, reference_source: str, target_source: str) -> None:
        ...

    def disable_auto_sync(self) -> None:
        ...

    def get_meters(self) -> dict:
        ...

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


class AudioRuntimeFactory(Protocol):
    def create(
        self,
        *,
        format: AudioFormat,
        output_queue: Any,
        tap_queue: Optional[Any] = None,
    ) -> AudioRuntimePort:
        ...
