from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


EventSink = Callable[[str, Dict[str, Any]], None]
MAX_PENDING_SPEAKER_UPDATES = 200


@dataclass
class HeadlessSource:
    name: str
    kind: str
    label: str
    enabled: bool = True
    delay_ms: float = 0.0

