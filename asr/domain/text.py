from __future__ import annotations

from typing import Tuple


def normalize_text(text: str) -> str:
    text = (text or "").strip()
    return " ".join(text.split())


def trim_overlap(prev_text: str, cur_text: str, *, max_window: int = 80, min_match: int = 8) -> Tuple[str, int]:
    prev = normalize_text(prev_text)
    cur = normalize_text(cur_text)
    if not prev or not cur:
        return (cur, 0)

    prev_tail = prev[-max_window:]
    best = 0
    max_k = min(len(prev_tail), len(cur))
    for k in range(min(max_k, max_window), min_match - 1, -1):
        if prev_tail[-k:] == cur[:k]:
            best = k
            break
    if best > 0:
        trimmed = cur[best:].lstrip()
        return (trimmed, best)
    return (cur, 0)
