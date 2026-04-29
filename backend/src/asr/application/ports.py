from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol


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
    def transcribe(self, audio_16k_mono: Any, *, beam_size: Optional[int] = None) -> Dict[str, Any]:
        ...


class AsrBackendFactoryPort(Protocol):
    def __call__(
        self,
        *,
        model_name: str,
        language: Optional[str],
        device: str,
        compute_type: str,
        cpu_threads: int,
        num_workers: int,
        beam_size: int,
        initial_prompt: Optional[str],
    ) -> AsrBackendPort:
        ...


class AudioSegmenterFactoryPort(Protocol):
    def __call__(
        self,
        *,
        config: Any,
        segment_queue: Any,
        diarization_queue: Optional[Any],
        diarization: Any,
        metrics: Any,
        log_event: Callable[[dict], None],
        segmentation_params: Callable[[], tuple[float, float, float]],
    ) -> Any:
        ...

class StopSignalPort(Protocol):
    def clear(self) -> None:
        ...

    def set(self) -> None:
        ...

    def is_set(self) -> bool:
        ...


class WorkerHandlePort(Protocol):
    def join(self, timeout: Optional[float] = None) -> None:
        ...


class RealtimeWorkerRunnerPort(Protocol):
    def create_stop_signal(self) -> StopSignalPort:
        ...

    def start_worker(self, *, name: str, target: Callable[[], None]) -> WorkerHandlePort:
        ...
