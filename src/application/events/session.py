from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from application.events.base import EventType, TypedEvent, init_event


@dataclass(frozen=True)
class AsrStopDoneEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = (
        "wav_path",
        "run_offline_pass",
        "offline_model_name",
        "offline_language",
        "stop_error",
    )
    wav_path: str
    run_offline_pass: bool
    offline_model_name: str
    offline_language: Optional[str]
    stop_error: Optional[str]

    def __init__(
        self,
        *,
        wav_path: str,
        run_offline_pass: bool,
        offline_model_name: str,
        offline_language: Optional[str],
        stop_error: Optional[str],
        ts: Optional[float] = None,
    ):
        init_event(
            self,
            EventType.ASR_STOP_DONE,
            ts,
            wav_path=str(wav_path),
            run_offline_pass=bool(run_offline_pass),
            offline_model_name=str(offline_model_name),
            offline_language=offline_language,
            stop_error=stop_error,
        )


@dataclass(frozen=True)
class OfflinePassStartedEvent(TypedEvent):
    def __init__(self, *, ts: Optional[float] = None):
        init_event(self, EventType.OFFLINE_PASS_STARTED, ts)


@dataclass(frozen=True)
class OfflinePassDoneEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("out_txt",)
    out_txt: str

    def __init__(self, *, out_txt: str, ts: Optional[float] = None):
        init_event(self, EventType.OFFLINE_PASS_DONE, ts, out_txt=str(out_txt))


@dataclass(frozen=True)
class OfflinePassErrorEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("error",)
    error: str

    def __init__(self, *, error: str, ts: Optional[float] = None):
        init_event(self, EventType.OFFLINE_PASS_ERROR, ts, error=str(error))
