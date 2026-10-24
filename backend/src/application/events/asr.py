from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from application.events.base import EventType, TypedEvent, init_event


@dataclass(frozen=True)
class SourceErrorEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ('source', 'error')
    source: str
    error: str

    def __init__(self, *, source: str, error: str, ts: float | None = None):
        init_event(
            self,
            EventType.SOURCE_ERROR,
            ts,
            source=str(source),
            error=str(error),
        )


@dataclass(frozen=True)
class UtteranceEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = (
        'text',
        'stream',
        'speaker',
        't_start',
        't_end',
        'overload',
    )
    text: str
    stream: str
    speaker: str = ''
    t_start: float | None = None
    t_end: float | None = None
    overload: bool = False

    def __init__(
        self,
        *,
        text: str,
        stream: str = '',
        speaker: str = '',
        t_start: float | None = None,
        t_end: float | None = None,
        overload: bool = False,
        ts: float | None = None,
    ):
        init_event(
            self,
            EventType.UTTERANCE,
            ts,
            text=str(text),
            stream=str(stream),
            speaker=str(speaker),
            t_start=t_start,
            t_end=t_end,
            overload=bool(overload),
        )


@dataclass(frozen=True)
class TranscriptSpeakerUpdateEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = (
        'line_id',
        'stream',
        'speaker',
        't_start',
        't_end',
        'confidence',
        'source',
    )
    line_id: str
    stream: str
    speaker: str
    t_start: float | None = None
    t_end: float | None = None
    confidence: float | None = None
    source: str = ''

    def __init__(
        self,
        *,
        speaker: str,
        line_id: str = '',
        stream: str = '',
        t_start: float | None = None,
        t_end: float | None = None,
        confidence: float | None = None,
        source: str = '',
        ts: float | None = None,
    ):
        init_event(
            self,
            EventType.TRANSCRIPT_SPEAKER_UPDATE,
            ts,
            line_id=str(line_id),
            stream=str(stream),
            speaker=str(speaker),
            t_start=t_start,
            t_end=t_end,
            confidence=confidence,
            source=str(source),
        )


@dataclass(frozen=True)
class AsrOverloadEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = (
        'active',
        'reason',
        'seg_qsize',
        'beam_cur',
        'lag_s',
    )
    active: bool
    reason: str
    seg_qsize: int | None
    beam_cur: int | None
    lag_s: float | None

    def __init__(
        self,
        *,
        active: bool,
        reason: str,
        seg_qsize: int | None = None,
        beam_cur: int | None = None,
        lag_s: float | None = None,
        ts: float | None = None,
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
    record_fields: ClassVar[tuple[str, ...]] = ('stream', 'reason', 'seg_qsize')
    stream: str
    reason: str
    seg_qsize: int | None

    def __init__(
        self,
        *,
        stream: str = '',
        reason: str = '',
        seg_qsize: int | None = None,
        ts: float | None = None,
    ):
        init_event(
            self,
            EventType.SEGMENT_DROPPED,
            ts,
            stream=str(stream),
            reason=str(reason),
            seg_qsize=seg_qsize,
        )


@dataclass(frozen=True)
class SegmentSkippedOverloadEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ('count', 'seg_qsize')
    count: int
    seg_qsize: int | None

    def __init__(
        self,
        *,
        count: int,
        seg_qsize: int | None = None,
        ts: float | None = None,
    ):
        init_event(
            self,
            EventType.SEGMENT_SKIPPED_OVERLOAD,
            ts,
            count=int(count),
            seg_qsize=seg_qsize,
        )


@dataclass(frozen=True)
class AsrMetricsEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = (
        'seg_dropped_total',
        'seg_skipped_total',
        'avg_latency_s',
        'p95_latency_s',
        'lag_s',
        'queue_wait_s',
        'rtf',
        'seg_qsize',
        'device',
    )
    seg_dropped_total: int
    seg_skipped_total: int
    avg_latency_s: float
    p95_latency_s: float
    lag_s: float
    queue_wait_s: float
    rtf: float
    seg_qsize: int
    device: str

    def __init__(
        self,
        *,
        seg_dropped_total: int = 0,
        seg_skipped_total: int = 0,
        avg_latency_s: float = 0.0,
        p95_latency_s: float = 0.0,
        lag_s: float = 0.0,
        queue_wait_s: float = 0.0,
        rtf: float = 0.0,
        seg_qsize: int = 0,
        device: str = '',
        ts: float | None = None,
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
            queue_wait_s=float(queue_wait_s),
            rtf=float(rtf),
            seg_qsize=int(seg_qsize),
            device=str(device),
        )


@dataclass(frozen=True)
class AsrInitStartEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ('model', 'device')
    model: str
    device: str

    def __init__(
        self, *, model: str = '', device: str = '', ts: float | None = None
    ):
        init_event(
            self,
            EventType.ASR_INIT_START,
            ts,
            model=str(model),
            device=str(device),
        )


@dataclass(frozen=True)
class AsrStartedEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = (
        'model',
        'mode',
        'language',
        'overload_strategy',
        'device',
        'compute_type',
        'cpu_threads',
        'num_workers',
        'beam_size',
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
        model: str = '',
        mode: str = '',
        language: str = '',
        overload_strategy: str = '',
        device: str = '',
        compute_type: str = '',
        cpu_threads: int = 0,
        num_workers: int = 1,
        beam_size: int = 0,
        ts: float | None = None,
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
    record_fields: ClassVar[tuple[str, ...]] = ('model',)
    model: str

    def __init__(self, *, model: str = '', ts: float | None = None):
        init_event(self, EventType.ASR_INIT_OK, ts, model=str(model))


@dataclass(frozen=True)
class AsrErrorEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ('component', 'where', 'error')
    component: str
    where: str
    error: str

    def __init__(
        self,
        *,
        where: str = '',
        error: str = '',
        component: str = 'asr',
        ts: float | None = None,
    ):
        init_event(
            self,
            EventType.ERROR,
            ts,
            component=str(component or 'asr'),
            where=str(where),
            error=str(error),
        )


@dataclass(frozen=True)
class AsrStoppedEvent(TypedEvent):
    def __init__(self, *, ts: float | None = None):
        init_event(self, EventType.ASR_STOPPED, ts)
