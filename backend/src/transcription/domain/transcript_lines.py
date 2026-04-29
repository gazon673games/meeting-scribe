from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def build_transcript_line_id(*, stream: str, t_start: Optional[float], t_end: Optional[float], ts: float) -> str:
    stream_key = _clean_part(stream or "mix")
    start_ms = _time_ms(t_start)
    end_ms = _time_ms(t_end)
    ts_ms = _time_ms(ts) or 0
    return f"{stream_key}:{start_ms or ts_ms}:{end_ms or ts_ms}"


def best_line_for_speaker_update(
    lines: Iterable[Dict[str, Any]],
    *,
    line_id: str = "",
    stream: str = "",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    candidates = list(lines)
    wanted_id = str(line_id or "").strip()
    if wanted_id:
        for line in candidates:
            if str(line.get("id", "")).strip() == wanted_id:
                return line

    best_line: Optional[Dict[str, Any]] = None
    best_overlap = 0.0
    for line in candidates:
        if stream and str(line.get("stream", "")) != str(stream):
            continue
        overlap = _overlap_seconds(
            _optional_float(line.get("t_start")),
            _optional_float(line.get("t_end")),
            t_start,
            t_end,
        )
        if overlap > best_overlap:
            best_line = line
            best_overlap = overlap
    return best_line if best_overlap > 0 else None


def update_line_speaker(
    line: Dict[str, Any],
    *,
    speaker: str,
    speaker_source: str = "",
    confidence: Optional[float] = None,
) -> bool:
    label = str(speaker or "").strip()
    if not label or label == "S?":
        return False
    if str(line.get("speaker", "")).strip() == label:
        return False
    line["speaker"] = label
    if speaker_source:
        line["speakerSource"] = str(speaker_source)
    if confidence is not None:
        line["speakerConfidence"] = float(confidence)
    return True


def _clean_part(value: str) -> str:
    out = []
    for char in str(value):
        out.append(char if char.isalnum() or char in {"_", "-"} else "_")
    return "".join(out).strip("_") or "mix"


def _time_ms(value: Optional[float]) -> Optional[int]:
    if value is None:
        return None
    return int(round(max(0.0, float(value)) * 1000.0))


def _optional_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _overlap_seconds(
    a_start: Optional[float],
    a_end: Optional[float],
    b_start: Optional[float],
    b_end: Optional[float],
) -> float:
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return 0.0
    return max(0.0, min(float(a_end), float(b_end)) - max(float(a_start), float(b_start)))
