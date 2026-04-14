from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from audio.application.source_controls import (
    reset_source_runtime_state,
    set_source_delay,
    set_source_enabled_state,
)
from audio.application.source_state import SourceState
from audio.domain.formats import AudioFormat
from audio.domain.ports import AudioFilter, AudioSource


class SourceRegistry:
    def __init__(self, format: AudioFormat) -> None:
        self._fmt = format
        self._sources: List[AudioSource] = []
        self._state: Dict[str, SourceState] = {}
        self._master_filters: List[AudioFilter] = []

    @property
    def state(self) -> Dict[str, SourceState]:
        return self._state

    @property
    def sources(self) -> List[AudioSource]:
        return self._sources

    def has_sources(self) -> bool:
        return bool(self._sources)

    def add_source(self, source: AudioSource, *, running: bool) -> None:
        if running:
            raise RuntimeError("Cannot add sources while engine is running (MVP).")
        if source.name in self._state:
            raise ValueError(f"Source '{source.name}' already exists")

        self._sources.append(source)
        state = SourceState(src=source)
        state.src_rate = int(source.get_format().sample_rate)
        self._state[source.name] = state

    def add_master_filter(self, flt: AudioFilter) -> None:
        self._master_filters.append(flt)

    def get_state(self, name: str) -> Optional[SourceState]:
        return self._state.get(name)

    def set_source_enabled(self, name: str, enabled: bool) -> None:
        state = self.get_state(name)
        if state is not None:
            set_source_enabled_state(state, enabled)

    def set_source_delay_ms(self, name: str, delay_ms: float) -> None:
        state = self.get_state(name)
        if state is not None:
            set_source_delay(state, self._fmt, delay_ms)

    def reset_runtime_state(self) -> None:
        for state in self._state.values():
            reset_source_runtime_state(state)

    def source_items(self) -> List[Tuple[str, SourceState]]:
        return list(self._state.items())

    def master_filters(self) -> List[AudioFilter]:
        return list(self._master_filters)
