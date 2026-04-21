from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Optional


class EventType(str, Enum):
    SOURCE_ERROR = "source_error"
    UTTERANCE = "utterance"
    ASR_OVERLOAD = "asr_overload"
    SEGMENT_DROPPED = "segment_dropped"
    SEGMENT_SKIPPED_OVERLOAD = "segment_skipped_overload"
    ASR_METRICS = "asr_metrics"
    ASR_INIT_START = "asr_init_start"
    ASR_STARTED = "asr_started"
    ASR_INIT_OK = "asr_init_ok"
    ERROR = "error"
    ASR_STOPPED = "asr_stopped"
    ASR_STOP_DONE = "asr_stop_done"
    OFFLINE_PASS_STARTED = "offline_pass_started"
    OFFLINE_PASS_DONE = "offline_pass_done"
    OFFLINE_PASS_ERROR = "offline_pass_error"
    CODEX_RESULT = "codex_result"
    CODEX_FALLBACK_STARTED = "codex_fallback_started"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TypedEvent:
    event_type: EventType
    ts: float

    record_fields: ClassVar[tuple[str, ...]] = ()

    def as_record(self) -> Dict[str, Any]:
        rec = {"type": self.event_type.value, "ts": float(self.ts)}
        rec.update({name: getattr(self, name) for name in self.record_fields})
        return rec


@dataclass(frozen=True)
class UnknownEvent(TypedEvent):
    record: Dict[str, Any]

    def __init__(self, record: Dict[str, Any]):
        init_event(self, EventType.UNKNOWN, safe_float(record.get("ts"), time.time()), record=dict(record))

    def as_record(self) -> Dict[str, Any]:
        return dict(self.record)


def event_to_record(event: Any) -> Dict[str, Any]:
    if isinstance(event, TypedEvent):
        return event.as_record()
    if isinstance(event, dict):
        return dict(event)
    return {"type": "unknown", "value": repr(event), "ts": time.time()}


def init_event(obj: TypedEvent, event_type: EventType, ts: Optional[float], **fields: Any) -> None:
    object.__setattr__(obj, "event_type", event_type)
    object.__setattr__(obj, "ts", time.time() if ts is None else float(ts))
    for key, value in fields.items():
        object.__setattr__(obj, key, value)


def safe_int(raw: Any, default: int) -> int:
    try:
        return int(raw)
    except Exception:
        return int(default)


def safe_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def optional_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    return safe_int(raw, 0)


def optional_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    return safe_float(raw, 0.0)
