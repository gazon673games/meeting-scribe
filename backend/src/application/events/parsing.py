from __future__ import annotations

import time
from typing import Any, Callable, Dict, Type

from application.events.asr import (
    AsrErrorEvent,
    AsrInitOkEvent,
    AsrInitStartEvent,
    AsrMetricsEvent,
    AsrOverloadEvent,
    AsrStartedEvent,
    AsrStoppedEvent,
    SegmentDroppedEvent,
    SegmentSkippedOverloadEvent,
    SourceErrorEvent,
    UtteranceEvent,
)
from application.events.assistant import CodexFallbackStartedEvent, CodexResultEvent
from application.events.base import EventType, TypedEvent, UnknownEvent, optional_float, optional_int, safe_float, safe_int
from application.events.session import AsrStopDoneEvent, OfflinePassDoneEvent, OfflinePassErrorEvent, OfflinePassStartedEvent

Converter = Callable[[Any], Any]
EventSpec = tuple[Type[TypedEvent], tuple[tuple[str, Converter, Any], ...]]


def event_from_record(raw: Any) -> TypedEvent:
    if isinstance(raw, TypedEvent):
        return raw
    if not isinstance(raw, dict):
        return UnknownEvent({"type": "unknown", "value": repr(raw), "ts": time.time()})

    typ = str(raw.get("type", ""))
    ts = safe_float(raw.get("ts"), time.time())
    spec = EVENT_SPECS.get(typ)
    if spec is None:
        return UnknownEvent(raw)
    event_cls, fields = spec
    kwargs = {name: converter(raw.get(name, default)) for name, converter, default in fields}
    return event_cls(ts=ts, **kwargs)


def _identity(raw: Any) -> Any:
    return raw


def _str(raw: Any) -> str:
    return str(raw)


def _bool(raw: Any) -> bool:
    return bool(raw)


def _int0(raw: Any) -> int:
    return safe_int(raw, 0)


def _float0(raw: Any) -> float:
    return safe_float(raw, 0.0)


EVENT_SPECS: Dict[str, EventSpec] = {
    EventType.SOURCE_ERROR.value: (SourceErrorEvent, (("source", _str, ""), ("error", _str, ""))),
    EventType.UTTERANCE.value: (
        UtteranceEvent,
        (("text", _str, ""), ("stream", _str, ""), ("overload", _bool, False)),
    ),
    EventType.ASR_OVERLOAD.value: (
        AsrOverloadEvent,
        (
            ("active", _bool, False),
            ("reason", _str, ""),
            ("seg_qsize", optional_int, None),
            ("beam_cur", optional_int, None),
            ("lag_s", optional_float, None),
        ),
    ),
    EventType.SEGMENT_DROPPED.value: (
        SegmentDroppedEvent,
        (("stream", _str, ""), ("reason", _str, ""), ("seg_qsize", optional_int, None)),
    ),
    EventType.SEGMENT_SKIPPED_OVERLOAD.value: (
        SegmentSkippedOverloadEvent,
        (("count", _int0, 0), ("seg_qsize", optional_int, None)),
    ),
    EventType.ASR_METRICS.value: (
        AsrMetricsEvent,
        (
            ("seg_dropped_total", _int0, 0),
            ("seg_skipped_total", _int0, 0),
            ("avg_latency_s", _float0, 0.0),
            ("p95_latency_s", _float0, 0.0),
            ("lag_s", _float0, 0.0),
        ),
    ),
    EventType.ASR_INIT_START.value: (AsrInitStartEvent, (("model", _str, ""), ("device", _str, ""))),
    EventType.ASR_STARTED.value: (
        AsrStartedEvent,
        (
            ("model", _str, ""),
            ("mode", _str, ""),
            ("language", _str, ""),
            ("overload_strategy", _str, ""),
            ("device", _str, ""),
            ("compute_type", _str, ""),
            ("cpu_threads", _int0, 0),
            ("num_workers", _int0, 1),
            ("beam_size", _int0, 0),
        ),
    ),
    EventType.ASR_INIT_OK.value: (AsrInitOkEvent, (("model", _str, ""),)),
    EventType.ERROR.value: (
        AsrErrorEvent,
        (("component", _str, "asr"), ("where", _str, ""), ("error", _str, "")),
    ),
    EventType.ASR_STOPPED.value: (AsrStoppedEvent, ()),
    EventType.ASR_STOP_DONE.value: (
        AsrStopDoneEvent,
        (
            ("wav_path", _str, ""),
            ("run_offline_pass", _bool, False),
            ("offline_model_name", _str, ""),
            ("offline_language", _identity, None),
            ("stop_error", _identity, None),
        ),
    ),
    EventType.OFFLINE_PASS_STARTED.value: (OfflinePassStartedEvent, ()),
    EventType.OFFLINE_PASS_DONE.value: (OfflinePassDoneEvent, (("out_txt", _str, ""),)),
    EventType.OFFLINE_PASS_ERROR.value: (OfflinePassErrorEvent, (("error", _str, ""),)),
    EventType.CODEX_RESULT.value: (
        CodexResultEvent,
        (("ok", _bool, False), ("profile", _str, ""), ("cmd", _str, ""), ("text", _str, ""), ("dt_s", _float0, 0.0)),
    ),
    EventType.CODEX_FALLBACK_STARTED.value: (
        CodexFallbackStartedEvent,
        (("profile", _str, ""), ("cmd", _str, ""), ("reason", _str, "")),
    ),
}
