from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Set

from audio.domain.types import TapMode


@dataclass(frozen=True)
class TapConfig:
    mode: TapMode = "both"
    sources_filter: Optional[Set[str]] = None
    drop_threshold: float = 0.85


def normalize_tap_config(
    *,
    mode: TapMode = "both",
    sources: Optional[Iterable[str]] = None,
    drop_threshold: float = 0.85,
) -> TapConfig:
    raw_mode = str(mode).strip().lower()
    normalized_mode: TapMode
    if raw_mode == "mix":
        normalized_mode = "mix"
    elif raw_mode == "sources":
        normalized_mode = "sources"
    else:
        normalized_mode = "both"

    threshold = max(0.1, min(0.99, float(drop_threshold)))
    source_filter = set(sources or ())
    return TapConfig(
        mode=normalized_mode,
        sources_filter=source_filter or None,
        drop_threshold=threshold,
    )
