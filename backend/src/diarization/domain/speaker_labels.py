from __future__ import annotations

from typing import Mapping


UNKNOWN_SPEAKER_LABEL = "S?"


def clean_speaker_label(value: object) -> str:
    label = str(value or "").strip()
    if not label or label == UNKNOWN_SPEAKER_LABEL:
        return ""
    return label


def source_speaker_label(
    labels_by_stream: Mapping[str, str] | None,
    stream: str,
    *,
    default: str = UNKNOWN_SPEAKER_LABEL,
) -> str:
    if not labels_by_stream:
        return default
    label = clean_speaker_label(labels_by_stream.get(str(stream), ""))
    return label or default
