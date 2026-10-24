from __future__ import annotations

from collections.abc import Iterable
from typing import Any

MIN_SPEAKER_UPDATE_EVENT_OVERLAP_RATIO = 0.5


def build_transcript_line_id(
    *, stream: str, t_start: float | None, t_end: float | None, ts: float
) -> str:
    stream_key = _clean_part(stream or 'mix')
    start_ms = _time_ms(t_start)
    end_ms = _time_ms(t_end)
    ts_ms = _time_ms(ts) or 0
    start = start_ms if start_ms is not None else ts_ms
    end = end_ms if end_ms is not None else ts_ms
    return f'{stream_key}:{start}:{end}'


def best_line_for_speaker_update(
    lines: Iterable[dict[str, Any]],
    *,
    line_id: str = '',
    stream: str = '',
    t_start: float | None = None,
    t_end: float | None = None,
) -> dict[str, Any] | None:
    candidates = list(lines)
    wanted_id = str(line_id or '').strip()
    if wanted_id:
        for line in candidates:
            if str(line.get('id', '')).strip() == wanted_id:
                return line

    best_line: dict[str, Any] | None = None
    best_overlap = 0.0
    for line in candidates:
        if stream and str(line.get('stream', '')) != str(stream):
            continue
        line_start = _optional_float(line.get('t_start'))
        line_end = _optional_float(line.get('t_end'))
        overlap = _overlap_seconds(
            line_start,
            line_end,
            t_start,
            t_end,
        )
        if not _speaker_update_overlap_is_meaningful(overlap, t_start, t_end):
            continue
        if overlap > best_overlap:
            best_line = line
            best_overlap = overlap
    return best_line


def update_line_speaker(
    line: dict[str, Any],
    *,
    speaker: str,
    speaker_source: str = '',
    confidence: float | None = None,
) -> bool:
    label = str(speaker or '').strip()
    if not label or label == 'S?':
        return False
    if str(line.get('speaker', '')).strip() == label:
        return False
    line['speaker'] = label
    if speaker_source:
        line['speakerSource'] = str(speaker_source)
    if confidence is not None:
        line['speakerConfidence'] = float(confidence)
    return True


def _clean_part(value: str) -> str:
    return (
        ''.join(
            char if char.isalnum() or char in {'_', '-'} else '_'
            for char in str(value)
        ).strip('_')
        or 'mix'
    )


def _time_ms(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(max(0.0, float(value)) * 1000.0))


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _overlap_seconds(
    a_start: float | None,
    a_end: float | None,
    b_start: float | None,
    b_end: float | None,
) -> float:
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return 0.0
    return max(
        0.0,
        min(float(a_end), float(b_end)) - max(float(a_start), float(b_start)),
    )


def _speaker_update_overlap_is_meaningful(
    overlap: float,
    event_start: float | None,
    event_end: float | None,
) -> bool:
    if overlap <= 0.0 or event_start is None or event_end is None:
        return False
    event_duration = max(0.0, float(event_end) - float(event_start))
    if event_duration <= 0.0:
        return False
    return (
        float(overlap) / event_duration
    ) >= MIN_SPEAKER_UPDATE_EVENT_OVERLAP_RATIO
