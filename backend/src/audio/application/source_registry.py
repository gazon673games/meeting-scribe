from __future__ import annotations

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
        self._sources: list[AudioSource] = []
        self._state: dict[str, SourceState] = {}
        self._master_filters: list[AudioFilter] = []

    @property
    def state(self) -> dict[str, SourceState]:
        return dict(self._state)

    @property
    def sources(self) -> list[AudioSource]:
        return list(self._sources)

    def has_sources(self) -> bool:
        return bool(self._sources)

    def add_source(self, source: AudioSource, *, running: bool) -> None:
        source_name = source.name
        if source_name in self._state:
            raise ValueError(f"Source '{source_name}' already exists")

        state = SourceState(
            src=source, src_rate=int(source.get_format().sample_rate)
        )
        self._sources.append(source)
        self._state[source_name] = state

    def remove_source(self, name: str, *, running: bool) -> AudioSource:
        if running:
            raise RuntimeError('Cannot remove sources while engine is running.')
        source_name = str(name)
        state = self._state.pop(source_name, None)
        if state is None:
            raise KeyError(f"Source '{source_name}' does not exist")
        self._sources = [
            source for source in self._sources if source.name != source_name
        ]
        return state.src

    def add_master_filter(self, flt: AudioFilter) -> None:
        self._master_filters.append(flt)

    def get_state(self, name: str) -> SourceState | None:
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

    def source_items(self) -> list[tuple[str, SourceState]]:
        return list(self._state.items())

    def master_filters(self) -> list[AudioFilter]:
        return list(self._master_filters)
