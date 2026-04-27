from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from application.events.base import EventType, TypedEvent, init_event


@dataclass(frozen=True)
class SourceErrorEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("source", "error")
    source: str
    error: str

    def __init__(self, *, source: str, error: str, ts: Optional[float] = None):
        init_event(self, EventType.SOURCE_ERROR, ts, source=str(source), error=str(error))


@dataclass(frozen=True)
class UtteranceEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("text", "stream", "overload")
    text: str
    stream: str
    overload: bool = False

    def __init__(self, *, text: str, stream: str = "", overload: bool = False, ts: Optional[float] = None):
        init_event(self, EventType.UTTERANCE, ts, text=str(text), stream=str(stream), overload=bool(overload))


@dataclass(frozen=True)
class AsrOverloadEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("active", "reason", "seg_qsize", "beam_cur", "lag_s")
    active: bool
    reason: str
    seg_qsize: Optional[int]
    beam_cur: Optional[int]
    lag_s: Optional[float]

    def __init__(
        self,
        *,
        active: bool,
        reason: str,
        seg_qsize: Optional[int] = None,
        beam_cur: Optional[int] = None,
        lag_s: Optional[float] = None,
        ts: Optional[float] = None,
    ):
        init_event(
            self,
            EventType.ASR_OVERLOAD,
            ts,
            active=bool(active),
            reason=str(reason),
            seg_qsize=seg_qsize,
            beam_cur=beam_cur,
            lag_s=lag_s,
        )


@dataclass(frozen=True)
class SegmentDroppedEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("stream", "reason", "seg_qsize")
    stream: str
    reason: str
    seg_qsize: Optional[int]

    def __init__(self, *, stream: str = "", reason: str = "", seg_qsize: Optional[int] = None, ts: Optional[float] = None):
        init_event(self, EventType.SEGMENT_DROPPED, ts, stream=str(stream), reason=str(reason), seg_qsize=seg_qsize)


@dataclass(frozen=True)
class SegmentSkippedOverloadEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("count", "seg_qsize")
    count: int
    seg_qsize: Optional[int]

    def __init__(self, *, count: int, seg_qsize: Optional[int] = None, ts: Optional[float] = None):
        init_event(self, EventType.SEGMENT_SKIPPED_OVERLOAD, ts, count=int(count), seg_qsize=seg_qsize)


@dataclass(frozen=True)
class AsrMetricsEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = (
        "seg_dropped_total",
        "seg_skipped_total",
        "avg_latency_s",
        "p95_latency_s",
        "lag_s",
    )
    seg_dropped_total: int
    seg_skipped_total: int
    avg_latency_s: float
    p95_latency_s: float
    lag_s: float

    def __init__(
        self,
        *,
        seg_dropped_total: int = 0,
        seg_skipped_total: int = 0,
        avg_latency_s: float = 0.0,
        p95_latency_s: float = 0.0,
        lag_s: float = 0.0,
        ts: Optional[float] = None,
    ):
        init_event(
            self,
            EventType.ASR_METRICS,
            ts,
            seg_dropped_total=int(seg_dropped_total),
            seg_skipped_total=int(seg_skipped_total),
            avg_latency_s=float(avg_latency_s),
            p95_latency_s=float(p95_latency_s),
            lag_s=float(lag_s),
        )


@dataclass(frozen=True)
class AsrInitStartEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("model", "device")
    model: str
    device: str

    def __init__(self, *, model: str = "", device: str = "", ts: Optional[float] = None):
        init_event(self, EventType.ASR_INIT_START, ts, model=str(model), device=str(device))


@dataclass(frozen=True)
class AsrStartedEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = (
        "model",
        "mode",
        "language",
        "overload_strategy",
        "device",
        "compute_type",
        "cpu_threads",
        "num_workers",
        "beam_size",
    )
    model: str
    mode: str
    language: str
    overload_strategy: str
    device: str
    compute_type: str
    cpu_threads: int
    num_workers: int
    beam_size: int

    def __init__(
        self,
        *,
        model: str = "",
        mode: str = "",
        language: str = "",
        overload_strategy: str = "",
        device: str = "",
        compute_type: str = "",
        cpu_threads: int = 0,
        num_workers: int = 1,
        beam_size: int = 0,
        ts: Optional[float] = None,
    ):
        init_event(
            self,
            EventType.ASR_STARTED,
            ts,
            model=str(model),
            mode=str(mode),
            language=str(language),
            overload_strategy=str(overload_strategy),
            device=str(device),
            compute_type=str(compute_type),
            cpu_threads=int(cpu_threads),
            num_workers=int(num_workers),
            beam_size=int(beam_size),
        )


@dataclass(frozen=True)
class AsrInitOkEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("model",)
    model: str

    def __init__(self, *, model: str = "", ts: Optional[float] = None):
        init_event(self, EventType.ASR_INIT_OK, ts, model=str(model))


@dataclass(frozen=True)
class AsrErrorEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("component", "where", "error")
    component: str
    where: str
    error: str

    def __init__(self, *, where: str = "", error: str = "", component: str = "asr", ts: Optional[float] = None):
        init_event(
            self,
            EventType.ERROR,
            ts,
            component=str(component or "asr"),
            where=str(where),
            error=str(error),
        )


@dataclass(frozen=True)
class AsrStoppedEvent(TypedEvent):
    def __init__(self, *, ts: Optional[float] = None):
        init_event(self, EventType.ASR_STOPPED, ts)
