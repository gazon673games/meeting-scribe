from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UtteranceState:
    stream: str
    speaker: str
    t_start: float
    t_end: float
    text: str
    last_emit_ts: float

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.t_end) - float(self.t_start))

    def can_extend(self, *, t_start: float, t_end: float, gap_s: float, max_s: float) -> bool:
        gap = float(t_start) - float(self.t_end)
        new_duration = float(t_end) - float(self.t_start)
        return gap <= float(gap_s) and new_duration <= float(max_s)

    def extend(self, *, t_end: float, text: str, last_emit_ts: float) -> None:
        self.t_end = float(t_end)
        self.text = (str(self.text).strip() + " " + str(text).strip()).strip()
        self.last_emit_ts = float(last_emit_ts)

    def should_flush(self, *, now: float, flush_s: float, force: bool) -> bool:
        return bool(force) or (float(now) - float(self.last_emit_ts)) >= float(flush_s)
