from __future__ import annotations

from typing import Any, Protocol

from diarization.domain.segments import DiarSegment


class OnlineDiarizerPort(Protocol):
    def last_error(self) -> str | None: ...

    def clusters_snapshot(self) -> list[dict]: ...

    def assign_with_debug(
        self, audio_16k: Any, ts: float | None = None
    ) -> tuple[str, int | None, float, bool]: ...


class OnlineDiarizerFactoryPort(Protocol):
    def __call__(
        self,
        *,
        similarity_threshold: float,
        min_segment_s: float,
        window_s: float,
        backend: str,
        device: str,
        temp_dir: Any | None = None,
        sherpa_model_path: str = '',
        sherpa_provider: str = 'cpu',
        sherpa_num_threads: int = 1,
    ) -> OnlineDiarizerPort: ...


class PyannoteDiarizerPort(Protocol):
    def diarize(
        self, audio_16k: Any, *, t_offset: float = 0.0
    ) -> list[DiarSegment]: ...


class PyannoteDiarizerFactoryPort(Protocol):
    def __call__(self, *, device: str) -> PyannoteDiarizerPort: ...
