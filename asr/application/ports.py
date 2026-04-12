from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from asr.domain import DiarSegment


class AsrLoggerPort(Protocol):
    max_bytes: int
    backup_count: int

    def write(self, rec: Dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...


class AsrLoggerFactoryPort(Protocol):
    def __call__(
        self,
        *,
        root: Any,
        session_id: str,
        language: str,
        max_bytes: int,
        backup_count: int,
    ) -> AsrLoggerPort:
        ...


class AsrBackendPort(Protocol):
    def transcribe(self, audio_16k_mono: np.ndarray, *, beam_size: Optional[int] = None) -> Dict[str, Any]:
        ...


class AsrBackendFactoryPort(Protocol):
    def __call__(
        self,
        *,
        model_name: str,
        language: Optional[str],
        device: str,
        compute_type: str,
        beam_size: int,
        initial_prompt: Optional[str],
    ) -> AsrBackendPort:
        ...


class OnlineDiarizerPort(Protocol):
    def last_error(self) -> Optional[str]:
        ...

    def assign_with_debug(
        self, audio_16k: np.ndarray, ts: Optional[float] = None
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
    ) -> OnlineDiarizerPort:
        ...


class PyannoteDiarizerPort(Protocol):
    def diarize(self, audio_16k: np.ndarray, *, t_offset: float = 0.0) -> List[DiarSegment]:
        ...


class PyannoteDiarizerFactoryPort(Protocol):
    def __call__(self, *, device: str) -> PyannoteDiarizerPort:
        ...
