from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from asr.domain.text import normalize_text, trim_overlap


@dataclass
class StreamDedupFilter:
    """Per-stream sliding-window text deduplication."""

    enabled: bool
    window: int
    min_match: int = 8
    _last: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def reset(self) -> None:
        self._last.clear()

    def filter(self, stream: str, text: str) -> Tuple[str, int]:
        """Return (deduplicated text, removed char count)."""
        if not self.enabled:
            if text:
                self._last[stream] = normalize_text(text)
            return text, 0

        prev = self._last.get(stream, "")
        trimmed, removed = trim_overlap(prev, text, max_window=self.window, min_match=self.min_match)
        if trimmed:
            self._last[stream] = normalize_text(prev + " " + trimmed).strip()
        return trimmed, removed
