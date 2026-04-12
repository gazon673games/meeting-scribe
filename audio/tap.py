from __future__ import annotations

import queue
from typing import Dict, Optional

import numpy as np

from audio.types import TapMode


def tap_should_send(
    tap_q: "queue.Queue[dict]",
    *,
    tap_queue_max: int,
    drop_threshold: float,
) -> bool:
    try:
        qsz = int(tap_q.qsize())
    except Exception:
        return True
    cap = max(1, int(tap_queue_max))
    return (qsz / float(cap)) < float(drop_threshold)


def build_tap_packet(
    *,
    t_start: float,
    t_end: float,
    mixed: np.ndarray,
    sources_out: Optional[Dict[str, np.ndarray]],
    mode: TapMode,
) -> Dict[str, object]:
    packet: Dict[str, object] = {"t_start": float(t_start), "t_end": float(t_end)}
    if mode in ("mix", "both"):
        packet["mix"] = np.array(mixed, copy=True)
    if mode in ("sources", "both"):
        packet["sources"] = {
            str(name): np.array(block, copy=True)
            for name, block in (sources_out or {}).items()
        }
    return packet
