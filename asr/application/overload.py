from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from asr.domain.types import OverloadStrategy


@dataclass
class OverloadController:
    enter_qsize: int
    exit_qsize: int
    hard_qsize: int
    hold_s: float
    beam_cap: int
    overlap_ms: float
    max_segment_s: float
    strategy: OverloadStrategy = "drop_old"
    hard_beam_cap: int = 1
    hard_overlap_ms: float = 80.0
    hard_max_segment_s: float = 3.5

    active: bool = False
    since_ts: float = 0.0
    last_event_ts: float = 0.0
    hard_active: bool = False
    hard_since_ts: float = 0.0

    def reset(self) -> None:
        self.active = False
        self.since_ts = 0.0
        self.last_event_ts = 0.0
        self.hard_active = False
        self.hard_since_ts = 0.0

    def update(self, *, seg_qsize: int, beam_cur: int, lag_s: float, now: float) -> List[dict]:
        events: List[dict] = []
        ev = self._update_regular(seg_qsize=seg_qsize, beam_cur=beam_cur, lag_s=lag_s, now=now)
        if ev is not None:
            events.append(ev)
        ev = self._update_hard(seg_qsize=seg_qsize, beam_cur=beam_cur, lag_s=lag_s, now=now)
        if ev is not None:
            events.append(ev)
        return events

    def segmentation_params(
        self, *, endpoint_silence_ms: float, max_segment_s: float, overlap_ms: float
    ) -> Tuple[float, float, float]:
        if self.active:
            if self.hard_active and self.strategy == "keep_all":
                return (
                    endpoint_silence_ms,
                    float(min(self.max_segment_s, self.hard_max_segment_s)),
                    float(min(self.overlap_ms, self.hard_overlap_ms)),
                )
            return (endpoint_silence_ms, self.max_segment_s, self.overlap_ms)
        return (endpoint_silence_ms, max_segment_s, overlap_ms)

    def limit_beam(self, beam: int) -> int:
        out = int(beam)
        if self.active:
            out = min(out, int(self.beam_cap))
        if self.hard_active and self.strategy == "keep_all":
            out = min(out, int(self.hard_beam_cap))
        return max(1, out)

    def drop_old_count(self, seg_qsize: int) -> int:
        if self.strategy != "drop_old":
            return 0
        if int(seg_qsize) < int(self.hard_qsize):
            return 0
        keep = max(1, int(self.exit_qsize))
        return max(0, int(seg_qsize) - keep)

    def _update_regular(self, *, seg_qsize: int, beam_cur: int, lag_s: float, now: float) -> dict | None:
        if not self.active:
            if int(seg_qsize) >= int(self.enter_qsize):
                self.active = True
                self.since_ts = now
                return self._maybe_event(
                    True,
                    f"enter: qsize>={self.enter_qsize}",
                    seg_qsize=seg_qsize,
                    beam_cur=beam_cur,
                    lag_s=lag_s,
                    now=now,
                )
            return None

        held = (now - float(self.since_ts)) < float(self.hold_s)
        if not held and int(seg_qsize) <= int(self.exit_qsize):
            self.active = False
            self.since_ts = 0.0
            return self._maybe_event(
                False,
                f"exit: qsize<={self.exit_qsize}",
                seg_qsize=seg_qsize,
                beam_cur=beam_cur,
                lag_s=lag_s,
                now=now,
            )
        return None

    def _update_hard(self, *, seg_qsize: int, beam_cur: int, lag_s: float, now: float) -> dict | None:
        if self.strategy != "keep_all":
            self.hard_active = False
            self.hard_since_ts = 0.0
            return None

        if not self.hard_active:
            if int(seg_qsize) >= int(self.hard_qsize):
                self.hard_active = True
                self.hard_since_ts = now
                return self._event(
                    True,
                    f"hard_overload_keep_all: qsize>={self.hard_qsize}",
                    seg_qsize=seg_qsize,
                    beam_cur=beam_cur,
                    lag_s=lag_s,
                    now=now,
                )
            return None

        if int(seg_qsize) <= int(self.exit_qsize):
            self.hard_active = False
            self.hard_since_ts = 0.0
            return self._event(
                True,
                f"hard_overload_keep_all_recover: qsize<={self.exit_qsize}",
                seg_qsize=seg_qsize,
                beam_cur=beam_cur,
                lag_s=lag_s,
                now=now,
            )
        return None

    def _maybe_event(
        self, active: bool, reason: str, *, seg_qsize: int, beam_cur: int, lag_s: float, now: float
    ) -> dict | None:
        if (now - float(self.last_event_ts)) < 0.6:
            return None
        self.last_event_ts = now
        return self._event(active, reason, seg_qsize=seg_qsize, beam_cur=beam_cur, lag_s=lag_s, now=now)

    @staticmethod
    def _event(active: bool, reason: str, *, seg_qsize: int, beam_cur: int, lag_s: float, now: float) -> dict:
        return {
            "type": "asr_overload",
            "active": bool(active),
            "reason": str(reason),
            "seg_qsize": int(seg_qsize),
            "beam_cur": int(beam_cur),
            "lag_s": float(lag_s),
            "ts": now,
        }
