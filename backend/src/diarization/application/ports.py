from __future__ import annotations

from typing import Any, List, Optional, Protocol

from diarization.domain.segments import DiarSegment


class OnlineDiarizerPort(Protocol):
    def last_error(self) -> Optional[str]:
        ...

    def assign_with_debug(
        self, audio_16k: Any, ts: Optional[float] = None
    ) -> tuple[str, Optional[int], float, bool]:
        ...


class OnlineDiarizerFactoryPort(Protocol):
    def __call__(
        self,
        *,
        similarity_threshold: float,
        min_segment_s: float,
        window_s: float,
        backend: str,
        device: str,
        temp_dir: Optional[Any] = None,
        sherpa_model_path: str = "",
        sherpa_provider: str = "cpu",
        sherpa_num_threads: int = 1,
    ) -> OnlineDiarizerPort:
        ...


class PyannoteDiarizerPort(Protocol):
    def diarize(self, audio_16k: Any, *, t_offset: float = 0.0) -> List[DiarSegment]:
        ...


class PyannoteDiarizerFactoryPort(Protocol):
    def __call__(self, *, device: str) -> PyannoteDiarizerPort:
        ...
