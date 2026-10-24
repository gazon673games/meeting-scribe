from __future__ import annotations

import logging
import math
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from interface.session_controller_parts.helpers import (
    default_source_name,
    normalize_source_kind,
    rms_to_pct,
    safe_float,
)
from interface.session_controller_parts.types import EventSink, HeadlessSource
from session.domain.speaker_labels import default_speaker_label_for_source_kind

_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from interface.session_controller import HeadlessSessionController


def set_event_sink(
    session: HeadlessSessionController, event_sink: EventSink | None
) -> None:
    with session._lock:
        session.event_sink = event_sink


def add_source(
    session: HeadlessSessionController,
    *,
    kind: str,
    token: object,
    label: str = '',
    name: str = '',
    device_id: str = '',
) -> dict[str, Any]:
    with session._lock:
        normalized_kind = normalize_source_kind(kind)
        source_name = session._make_unique_name(
            name or default_source_name(normalized_kind)
        )

        if normalized_kind == 'loopback':
            source = session.audio_source_factory.create_loopback_source(
                name=source_name,
                engine_format=session.format,
                device=token,
                error_callback=session._on_source_error,
            )
        elif normalized_kind == 'process':
            source = session.audio_source_factory.create_process_source(
                name=source_name,
                token=token,
                error_callback=session._on_source_error,
            )
        else:
            source = session.audio_source_factory.create_microphone_source(
                name=source_name, device=token
            )

        session.engine.add_source(source)
        session._source_objs[source_name] = source
        session._sources[source_name] = HeadlessSource(
            name=source_name,
            kind=normalized_kind,
            label=str(label or source_name),
            device_id=device_id,
        )
        session._status = f'added {source_name}'
        if session._session_state.is_running:
            session.engine.set_source_enabled(source_name, True)
            session.engine.set_source_delay_ms(source_name, 0.0)
            if session._asr_requested:
                session._apply_tap_config()
        session._emit(
            'source_added', {'source': session._source_record(source_name)}
        )
        return session._source_record(source_name)


def remove_source(
    session: HeadlessSessionController, *, name: str
) -> dict[str, Any]:
    with session._lock:
        session._ensure_not_running('remove a source')
        record = session._source_record(name)
        session.engine.remove_source(record['name'])
        session._source_objs.pop(record['name'], None)
        session._sources.pop(record['name'], None)
        session._status = f'removed {record["name"]}'
        session._emit('source_removed', {'source': record})
        return record


def set_source_enabled(
    session: HeadlessSessionController, *, name: str, enabled: bool
) -> dict[str, Any]:
    with session._lock:
        source = session._require_source(name)
        value = bool(enabled)
        session.engine.set_source_enabled(source.name, value)
        source.enabled = value
        if session._session_state.is_running and session._asr_requested:
            session._apply_tap_config()
        session._emit(
            'source_updated', {'source': session._source_record(source.name)}
        )
        return session._source_record(source.name)


def set_source_delay(
    session: HeadlessSessionController, *, name: str, delay_ms: object
) -> dict[str, Any]:
    with session._lock:
        source = session._require_source(name)
        value = safe_float(delay_ms, 0.0)
        if not math.isfinite(value):
            raise ValueError('Source delay must be finite')
        if value < 0.0:
            value = 0.0
        session.engine.set_source_delay_ms(source.name, value)
        source.delay_ms = value
        session._emit(
            'source_updated', {'source': session._source_record(source.name)}
        )
        return session._source_record(source.name)


def clear_transcript(session: HeadlessSessionController) -> dict[str, Any]:
    with session._lock:
        session._transcript_lines.clear()
        session._session_transcript_lines.clear()
        session._pending_speaker_updates.clear()
        session._emit('transcript_cleared', {})
        return session.snapshot()


def _safe_meters(session: HeadlessSessionController) -> dict[str, Any]:
    try:
        meters = session.engine.get_meters()
    except Exception as error:
        session._last_error = f'{type(error).__name__}: {error}'
        return {}
    return meters if isinstance(meters, dict) else {}


def _source_record(
    session: HeadlessSessionController,
    name: str,
    *,
    meters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = session._require_source(name)
    source_meters = ((meters or {}).get('sources', {}) or {}).get(name, {})
    rms = safe_float(source_meters.get('rms', 0.0), 0.0)
    last_ts = safe_float(source_meters.get('last_ts', 0.0), 0.0)
    active = bool(time.monotonic() - last_ts < 0.6 and rms > 1e-4)
    status = 'active' if active else 'silence'
    if not source.enabled:
        status = 'muted'
    return {
        'name': source.name,
        'kind': source.kind,
        'deviceId': source.device_id,
        'label': source.label,
        'enabled': bool(source.enabled),
        'delayMs': float(source.delay_ms),
        'rms': rms,
        'level': rms_to_pct(rms),
        'active': active,
        'status': status,
        'bufferFrames': int(
            safe_float(source_meters.get('buffer_frames', 0), 0.0)
        ),
        'droppedInFrames': int(
            safe_float(source_meters.get('dropped_in_frames', 0), 0.0)
        ),
        'missingOutFrames': int(
            safe_float(source_meters.get('missing_out_frames', 0), 0.0)
        ),
        'sampleRate': int(safe_float(source_meters.get('src_rate', 0), 0.0)),
    }


def _source_speaker_labels_locked(
    session: HeadlessSessionController,
) -> dict[str, str]:
    return {
        source.name: default_speaker_label_for_source_kind(source.kind)
        for source in session._sources.values()
        if default_speaker_label_for_source_kind(source.kind)
    }


def _speaker_label_for_stream_locked(
    session: HeadlessSessionController, stream: str
) -> str:
    source = session._sources.get(str(stream))
    if source is None:
        return ''
    return default_speaker_label_for_source_kind(source.kind)


def _require_source(
    session: HeadlessSessionController, name: str
) -> HeadlessSource:
    source = session._sources.get(str(name))
    if source is None:
        raise KeyError(f'Unknown source: {name}')
    return source


def _make_unique_name(session: HeadlessSessionController, base: str) -> str:
    candidate = str(base or 'source').strip() or 'source'
    if candidate not in session._sources:
        return candidate
    index = 2
    while f'{candidate}_{index}' in session._sources:
        index += 1
    return f'{candidate}_{index}'


def _ensure_not_running(
    session: HeadlessSessionController, action: str
) -> None:
    if session._session_state.is_running or session.engine.is_running():
        raise RuntimeError(f'Stop the current session before you {action}')


def _on_source_error(
    session: HeadlessSessionController, source: str, error: str
) -> None:
    message = f'{source}: {error}'
    with session._lock:
        session._last_error = message
    session._emit(
        'source_error', {'source': str(source), 'message': str(error)}
    )


def _emit(
    session: HeadlessSessionController, event_type: str, payload: dict[str, Any]
) -> None:
    sink = session.event_sink
    if sink is None:
        return
    try:
        sink(event_type, deepcopy({'ts': time.time(), **payload}))
    except Exception:
        _logger.debug('Best-effort operation failed in _emit', exc_info=True)


class SourceControlMixin:
    """Compatibility facade for callers using the former mixin API."""

    set_event_sink = set_event_sink
    add_source = add_source
    remove_source = remove_source
    set_source_enabled = set_source_enabled
    set_source_delay = set_source_delay
    clear_transcript = clear_transcript
    _safe_meters = _safe_meters
    _source_record = _source_record
    _source_speaker_labels_locked = _source_speaker_labels_locked
    _speaker_label_for_stream_locked = _speaker_label_for_stream_locked
    _require_source = _require_source
    _make_unique_name = _make_unique_name
    _ensure_not_running = _ensure_not_running
    _on_source_error = _on_source_error
    _emit = _emit
