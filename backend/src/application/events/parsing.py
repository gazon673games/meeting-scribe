from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

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
    TranscriptSpeakerUpdateEvent,
    UtteranceEvent,
)
from application.events.assistant import (
    CodexFallbackStartedEvent,
    CodexResultEvent,
)
from application.events.base import (
    EventType,
    TypedEvent,
    UnknownEvent,
    optional_float,
    optional_int,
    safe_float,
    safe_int,
)
from application.events.session import (
    AsrStopDoneEvent,
    OfflinePassDoneEvent,
    OfflinePassErrorEvent,
    OfflinePassStartedEvent,
)

Converter = Callable[[Any], Any]
EventSpec = tuple[type[TypedEvent], tuple[tuple[str, Converter, Any], ...]]


def event_from_record(raw: Any) -> TypedEvent:
    if isinstance(raw, TypedEvent):
        return raw
    if not isinstance(raw, dict):
        return UnknownEvent(
            {'type': 'unknown', 'value': repr(raw), 'ts': time.time()}
        )

    event_type = str(raw.get('type', ''))
    ts = safe_float(raw.get('ts'), time.time())
    spec = EVENT_SPECS.get(event_type)
    if spec is None:
        return UnknownEvent(raw)
    event_cls, fields = spec
    kwargs = {
        name: converter(raw.get(name, default))
        for name, converter, default in fields
    }
    return event_cls(ts=ts, **kwargs)


def _identity(raw: Any) -> Any:
    return raw


def _int0(raw: Any) -> int:
    return safe_int(raw, 0)


def _float0(raw: Any) -> float:
    return safe_float(raw, 0.0)


def _optional_str(raw: Any) -> str:
    return '' if raw is None else str(raw)


EVENT_SPECS: Mapping[str, EventSpec] = MappingProxyType(
    {
        EventType.SOURCE_ERROR.value: (
            SourceErrorEvent,
            (('source', str, ''), ('error', str, '')),
        ),
        EventType.UTTERANCE.value: (
            UtteranceEvent,
            (
                ('text', str, ''),
                ('stream', str, ''),
                ('speaker', str, ''),
                ('t_start', optional_float, None),
                ('t_end', optional_float, None),
                ('overload', bool, False),
            ),
        ),
        EventType.TRANSCRIPT_SPEAKER_UPDATE.value: (
            TranscriptSpeakerUpdateEvent,
            (
                ('line_id', _optional_str, ''),
                ('stream', _optional_str, ''),
                ('speaker', str, ''),
                ('t_start', optional_float, None),
                ('t_end', optional_float, None),
                ('confidence', optional_float, None),
                ('source', _optional_str, ''),
            ),
        ),
        EventType.ASR_OVERLOAD.value: (
            AsrOverloadEvent,
            (
                ('active', bool, False),
                ('reason', str, ''),
                ('seg_qsize', optional_int, None),
                ('beam_cur', optional_int, None),
                ('lag_s', optional_float, None),
            ),
        ),
        EventType.SEGMENT_DROPPED.value: (
            SegmentDroppedEvent,
            (
                ('stream', str, ''),
                ('reason', str, ''),
                ('seg_qsize', optional_int, None),
            ),
        ),
        EventType.SEGMENT_SKIPPED_OVERLOAD.value: (
            SegmentSkippedOverloadEvent,
            (('count', _int0, 0), ('seg_qsize', optional_int, None)),
        ),
        EventType.ASR_METRICS.value: (
            AsrMetricsEvent,
            (
                ('seg_dropped_total', _int0, 0),
                ('seg_skipped_total', _int0, 0),
                ('avg_latency_s', _float0, 0.0),
                ('p95_latency_s', _float0, 0.0),
                ('lag_s', _float0, 0.0),
                ('queue_wait_s', _float0, 0.0),
                ('rtf', _float0, 0.0),
                ('seg_qsize', _int0, 0),
                ('device', str, ''),
            ),
        ),
        EventType.ASR_INIT_START.value: (
            AsrInitStartEvent,
            (('model', str, ''), ('device', str, '')),
        ),
        EventType.ASR_STARTED.value: (
            AsrStartedEvent,
            (
                ('model', str, ''),
                ('mode', str, ''),
                ('language', str, ''),
                ('overload_strategy', str, ''),
                ('device', str, ''),
                ('compute_type', str, ''),
                ('cpu_threads', _int0, 0),
                ('num_workers', _int0, 1),
                ('beam_size', _int0, 0),
            ),
        ),
        EventType.ASR_INIT_OK.value: (AsrInitOkEvent, (('model', str, ''),)),
        EventType.ERROR.value: (
            AsrErrorEvent,
            (('component', str, 'asr'), ('where', str, ''), ('error', str, '')),
        ),
        EventType.ASR_STOPPED.value: (AsrStoppedEvent, ()),
        EventType.ASR_STOP_DONE.value: (
            AsrStopDoneEvent,
            (
                ('wav_path', str, ''),
                ('run_offline_pass', bool, False),
                ('offline_model_name', str, ''),
                ('offline_language', _identity, None),
                ('stop_error', _identity, None),
            ),
        ),
        EventType.OFFLINE_PASS_STARTED.value: (OfflinePassStartedEvent, ()),
        EventType.OFFLINE_PASS_DONE.value: (
            OfflinePassDoneEvent,
            (('out_txt', str, ''),),
        ),
        EventType.OFFLINE_PASS_ERROR.value: (
            OfflinePassErrorEvent,
            (('error', str, ''),),
        ),
        EventType.CODEX_RESULT.value: (
            CodexResultEvent,
            (
                ('ok', bool, False),
                ('profile', str, ''),
                ('cmd', str, ''),
                ('text', str, ''),
                ('dt_s', _float0, 0.0),
                ('provider', str, 'codex'),
                ('model', str, ''),
                ('error_code', str, ''),
                ('retryable', bool, False),
                ('suggestion', str, ''),
                ('details', str, ''),
            ),
        ),
        EventType.CODEX_FALLBACK_STARTED.value: (
            CodexFallbackStartedEvent,
            (('profile', str, ''), ('cmd', str, ''), ('reason', str, '')),
        ),
    }
)
