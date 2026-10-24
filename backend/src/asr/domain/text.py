from __future__ import annotations


def normalize_text(text: str) -> str:
    text = (text or '').strip()
    return ' '.join(text.split())


def trim_overlap(
    prev_text: str, cur_text: str, *, max_window: int = 80, min_match: int = 8
) -> tuple[str, int]:
    previous = normalize_text(prev_text)
    current = normalize_text(cur_text)
    if not previous or not current:
        return current, 0

    previous_tail = previous[-max_window:]
    maximum_overlap = min(len(previous_tail), len(current), max_window)
    for overlap_length in range(maximum_overlap, min_match - 1, -1):
        if overlap_length <= 0:
            break
        if previous_tail[-overlap_length:] == current[:overlap_length]:
            trimmed = current[overlap_length:].lstrip()
            return trimmed, overlap_length
    return current, 0
