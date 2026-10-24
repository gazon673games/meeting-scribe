from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from queue import Queue
from typing import Any, Protocol

import numpy as np

from asr.application.metrics import ASRMetrics
from asr.application.segmentation import AudioSegmenterPort, SegmenterConfig
from asr.domain.recognition import RecognitionResult, RecognizedWord
from asr.domain.segments import Segment
from diarization.application.diarization import DiarizationPort


class AsrLoggerPort(Protocol):
    max_bytes: int
    backup_count: int

    def write(self, rec: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class AsrLoggerFactoryPort(Protocol):
    def __call__(
        self,
        *,
        root: Path,
        session_id: str,
        language: str,
        max_bytes: int,
        backup_count: int,
    ) -> AsrLoggerPort: ...


class AsrBackendPort(Protocol):
    def transcribe(
        self, audio_16k_mono: np.ndarray, *, beam_size: int | None = None
    ) -> RecognitionResult: ...

    def close(self) -> None: ...


class AsrBackendFactoryPort(Protocol):
    def __call__(
        self,
        *,
        model_name: str,
        language: str | None,
        device: str,
        compute_type: str,
        cpu_threads: int,
        num_workers: int,
        beam_size: int,
        initial_prompt: str | None,
        hotwords: str | None,
    ) -> AsrBackendPort: ...


class AudioSegmenterFactoryPort(Protocol):
    def __call__(
        self,
        *,
        config: SegmenterConfig,
        segment_queue: Queue[Segment],
        diarization_queue: Queue[Segment] | None,
        diarization: DiarizationPort,
        metrics: ASRMetrics,
        log_event: Callable[[dict], None],
        segmentation_params: Callable[[], tuple[float, float, float]],
    ) -> AudioSegmenterPort: ...


class StreamingAsrBackendPort(Protocol):
    def transcribe_words(
        self, audio_16k_mono: np.ndarray
    ) -> list[RecognizedWord]: ...


class StatefulTranscribePort(Protocol):
    def __call__(
        self,
        stream_id: str,
        audio_16k_mono: np.ndarray,
        *,
        is_final: bool,
    ) -> RecognitionResult: ...


class StopSignalPort(Protocol):
    def clear(self) -> None: ...

    def set(self) -> None: ...

    def is_set(self) -> bool: ...


class WorkerHandlePort(Protocol):
    def join(self, timeout: float | None = None) -> None: ...


class RealtimeWorkerRunnerPort(Protocol):
    def create_stop_signal(self) -> StopSignalPort: ...

    def start_worker(
        self, *, name: str, target: Callable[[], None]
    ) -> WorkerHandlePort: ...
