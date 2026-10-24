from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from application.events import (
    AsrMetricsEvent,
    SourceErrorEvent,
    TranscriptSpeakerUpdateEvent,
    UtteranceEvent,
    event_from_record,
    event_to_record,
)
from interface.session_controller_parts.helpers import (
    clean_speaker,
    join_words,
    optional_float,
)
from interface.session_controller_parts.types import MAX_PENDING_SPEAKER_UPDATES
from shared.domain.mappings import first_param
from transcription.domain.transcript_lines import (
    best_line_for_speaker_update,
    build_transcript_line_id,
    update_line_speaker,
)

_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from interface.session_controller import HeadlessSessionController


def _configure_transcript_files_locked(
    session: HeadlessSessionController, params: dict[str, Any]
) -> None:
    session._realtime_transcript_enabled = bool(
        first_param(
            params,
            'realtimeTranscriptToFile',
            'rtTranscriptToFile',
            'rt_transcript_to_file',
            default=False,
        )
    )
    store = session.transcript_store
    if store is None:
        return
    try:
        store.set_realtime_enabled(session._realtime_transcript_enabled)
        if session._asr_running:
            session._human_log_path = store.open_human_log()
        else:
            session._human_log_path = None
        session._sync_transcript_store_paths_locked()
    except (OSError, ValueError):
        _logger.warning('Cannot configure transcript files', exc_info=True)
        session._human_log_path = None
        session._realtime_transcript_path = None


def _close_transcript_files_locked(session: HeadlessSessionController) -> None:
    store = session.transcript_store
    if store is None:
        session._human_log_path = None
        session._realtime_transcript_path = None
        return
    try:
        try:
            store.close_human_log()
        finally:
            store.close_realtime_transcript()
    finally:
        session._realtime_transcript_enabled = False
        session._sync_transcript_store_paths_locked()


def _write_transcript_line_locked(
    session: HeadlessSessionController, line: dict[str, Any]
) -> None:
    store = session.transcript_store
    if store is None:
        return
    text = str(line.get('text', '')).strip()
    if not text:
        return
    ts = _event_timestamp(line)
    stream = str(line.get('stream') or 'mix')
    speaker = clean_speaker(line.get('speaker', ''))
    label = speaker or stream
    formatted = (
        f'[{time.strftime("%H:%M:%S", time.localtime(ts))}] {label}: {text}'
    )
    try:
        store.write_human_line(formatted)
        if session._realtime_transcript_enabled:
            store.write_realtime_srt_entry(ts, stream, text, speaker=speaker)
        session._sync_transcript_store_paths_locked()
    except Exception:
        _logger.debug(
            'Best-effort operation failed in _write_transcript_line_locked',
            exc_info=True,
        )


def _sync_transcript_store_paths_locked(
    session: HeadlessSessionController,
) -> None:
    store = session.transcript_store
    if store is None:
        return
    session._human_log_path = store.current_human_log_path
    session._realtime_transcript_path = store.realtime_transcript_path


def _handle_asr_event(session: HeadlessSessionController, raw: object) -> None:
    event = event_from_record(raw)
    record = event_to_record(event)
    with session._lock:
        session._dispatch_asr_event(event, record)
    session._emit_asr_notifications(event, record)


def _dispatch_asr_event(
    session: HeadlessSessionController, event: object, record: dict[str, Any]
) -> None:
    if isinstance(event, UtteranceEvent):
        session._handle_utterance_locked(event, record)
    elif isinstance(event, TranscriptSpeakerUpdateEvent):
        session._handle_speaker_update_locked(event, record)
    elif isinstance(event, AsrMetricsEvent):
        session._handle_metrics_locked(event)
    elif isinstance(event, SourceErrorEvent):
        session._last_error = f'{event.source}: {event.error}'
    elif record.get('type') == 'streaming_words':
        session._handle_streaming_words_locked(record)
    elif record.get('type') == 'streaming_final':
        session._handle_streaming_final_locked(record)


def _handle_utterance_locked(
    session: HeadlessSessionController,
    event: UtteranceEvent,
    record: dict[str, Any],
) -> None:
    text = event.text.strip()
    if not text:
        return
    stream = str(event.stream)
    speaker = clean_speaker(
        event.speaker
    ) or session._speaker_label_for_stream_locked(stream)
    t_start = optional_float(event.t_start)
    t_end = optional_float(event.t_end)
    line = {
        'id': build_transcript_line_id(
            stream=stream, t_start=t_start, t_end=t_end, ts=float(event.ts)
        ),
        'ts': float(event.ts),
        'stream': stream,
        'speaker': speaker,
        't_start': t_start,
        't_end': t_end,
        'text': text,
        'overload': bool(event.overload),
    }
    session._apply_pending_speaker_updates_locked(line)
    session._append_transcript_line_locked(line)
    record['line'] = line
    session._write_transcript_line_locked(line)


def _handle_speaker_update_locked(
    session: HeadlessSessionController,
    event: TranscriptSpeakerUpdateEvent,
    record: dict[str, Any],
) -> None:
    line = session._find_line_for_speaker_update_locked(event)
    if line is None:
        session._remember_pending_speaker_update_locked(event)
    elif session._update_line_speaker_from_event_locked(line, event):
        record['line'] = dict(line)


def _handle_metrics_locked(
    session: HeadlessSessionController, event: AsrMetricsEvent
) -> None:
    session._asr_metrics = {
        'segDroppedTotal': int(event.seg_dropped_total),
        'segSkippedTotal': int(event.seg_skipped_total),
        'avgLatencyS': float(event.avg_latency_s),
        'p95LatencyS': float(event.p95_latency_s),
        'lagS': float(event.lag_s),
        'queueWaitS': float(event.queue_wait_s),
        'rtf': float(event.rtf),
        'queueSize': int(event.seg_qsize),
        'device': str(event.device),
    }


def _handle_streaming_words_locked(
    session: HeadlessSessionController, record: dict[str, Any]
) -> None:
    stream = str(record.get('stream') or 'mix')
    new_confirmed = join_words(record.get('confirmed'))
    tentative_text = join_words(record.get('tentative'))
    tentative_id = f'streaming-{stream}'
    existing = next(
        (
            line
            for line in session._transcript_lines
            if line.get('id') == tentative_id
        ),
        None,
    )
    if existing is not None:
        confirmed = ' '.join(
            filter(None, [str(existing.get('_confirmed', '')), new_confirmed])
        )
        existing['_confirmed'] = confirmed
        existing['text'] = ' '.join(filter(None, [confirmed, tentative_text]))
        existing['t_end'] = optional_float(record.get('t_end'))
    else:
        text = ' '.join(filter(None, [new_confirmed, tentative_text]))
        if text:
            session._append_transcript_line_locked(
                {
                    'id': tentative_id,
                    '_confirmed': new_confirmed,
                    'ts': _event_timestamp(record),
                    'stream': stream,
                    'speaker': '',
                    't_start': optional_float(record.get('t_start')),
                    't_end': optional_float(record.get('t_end')),
                    'text': text,
                    'overload': False,
                }
            )


def _handle_streaming_final_locked(
    session: HeadlessSessionController, record: dict[str, Any]
) -> None:
    stream = str(record.get('stream') or 'mix')
    tentative_id = f'streaming-{stream}'
    session._transcript_lines = [
        line
        for line in session._transcript_lines
        if line.get('id') != tentative_id
    ]
    text = join_words(record.get('words'))
    if not text:
        return
    t_start = optional_float(record.get('t_start'))
    t_end = optional_float(record.get('t_end'))
    ts = _event_timestamp(record)
    line = {
        'id': build_transcript_line_id(
            stream=stream, t_start=t_start, t_end=t_end, ts=ts
        ),
        'ts': ts,
        'stream': stream,
        'speaker': '',
        't_start': t_start,
        't_end': t_end,
        'text': text,
        'overload': False,
    }
    session._apply_pending_speaker_updates_locked(line)
    session._append_transcript_line_locked(line)
    session._write_transcript_line_locked(line)


def _append_transcript_line_locked(
    session: HeadlessSessionController, line: dict[str, Any]
) -> None:
    session._transcript_lines.append(line)
    if not str(line.get('id', '')).startswith('streaming-'):
        session._session_transcript_lines.append(line)
    if len(session._transcript_lines) > 500:
        del session._transcript_lines[:-500]


def _emit_asr_notifications(
    session: HeadlessSessionController, event: object, record: dict[str, Any]
) -> None:
    session._emit('asr_event', record)
    if (
        isinstance(event, UtteranceEvent)
        and str(record.get('text', '')).strip()
    ):
        session._emit('transcript_line', record.get('line', {}))
    if isinstance(event, TranscriptSpeakerUpdateEvent) and record.get('line'):
        session._emit('transcript_line_update', record.get('line', {}))


def _update_line_speaker_from_event_locked(
    session: HeadlessSessionController,
    line: dict[str, Any],
    event: TranscriptSpeakerUpdateEvent,
) -> bool:
    return update_line_speaker(
        line,
        speaker=str(event.speaker),
        speaker_source=str(event.source or 'diarization'),
        confidence=optional_float(event.confidence),
    )


def _find_line_for_speaker_update_locked(
    session: HeadlessSessionController,
    event: TranscriptSpeakerUpdateEvent,
    *,
    lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    return best_line_for_speaker_update(
        lines if lines is not None else session._session_transcript_lines,
        line_id=str(event.line_id or ''),
        stream=str(event.stream or ''),
        t_start=optional_float(event.t_start),
        t_end=optional_float(event.t_end),
    )


def _remember_pending_speaker_update_locked(
    session: HeadlessSessionController, event: TranscriptSpeakerUpdateEvent
) -> None:
    if not clean_speaker(event.speaker):
        return
    session._pending_speaker_updates.append(event)
    if len(session._pending_speaker_updates) > MAX_PENDING_SPEAKER_UPDATES:
        del session._pending_speaker_updates[:-MAX_PENDING_SPEAKER_UPDATES]


def _apply_pending_speaker_updates_locked(
    session: HeadlessSessionController, line: dict[str, Any]
) -> None:
    pending: list[TranscriptSpeakerUpdateEvent] = []
    for event in session._pending_speaker_updates:
        target = session._find_line_for_speaker_update_locked(
            event, lines=[line]
        )
        if target is line:
            session._update_line_speaker_from_event_locked(line, event)
        else:
            pending.append(event)
    session._pending_speaker_updates = pending[-MAX_PENDING_SPEAKER_UPDATES:]


def _event_timestamp(record: dict[str, Any]) -> float:
    value = record.get('ts')
    return time.time() if value is None else float(value)


class TranscriptPipelineMixin:
    """Compatibility facade for callers using the former mixin API."""

    _configure_transcript_files_locked = _configure_transcript_files_locked
    _close_transcript_files_locked = _close_transcript_files_locked
    _write_transcript_line_locked = _write_transcript_line_locked
    _sync_transcript_store_paths_locked = _sync_transcript_store_paths_locked
    _handle_asr_event = _handle_asr_event
    _dispatch_asr_event = _dispatch_asr_event
    _handle_utterance_locked = _handle_utterance_locked
    _handle_speaker_update_locked = _handle_speaker_update_locked
    _handle_metrics_locked = _handle_metrics_locked
    _handle_streaming_words_locked = _handle_streaming_words_locked
    _handle_streaming_final_locked = _handle_streaming_final_locked
    _append_transcript_line_locked = _append_transcript_line_locked
    _emit_asr_notifications = _emit_asr_notifications
    _update_line_speaker_from_event_locked = (
        _update_line_speaker_from_event_locked
    )
    _find_line_for_speaker_update_locked = _find_line_for_speaker_update_locked
    _remember_pending_speaker_update_locked = (
        _remember_pending_speaker_update_locked
    )
    _apply_pending_speaker_updates_locked = (
        _apply_pending_speaker_updates_locked
    )
