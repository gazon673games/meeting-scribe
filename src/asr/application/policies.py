from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from asr.application.pipeline_config import ASRPipelineSettings


@dataclass
class AdaptiveBeam:
    min_beam: int = 1
    max_beam: int = 5
    cur_beam: int = 5

    @classmethod
    def from_settings(cls, settings: ASRPipelineSettings) -> AdaptiveBeam:
        max_beam = settings.resolved_adaptive_beam_max
        return cls(
            min_beam=max(1, int(settings.adaptive_beam_min)),
            max_beam=max_beam,
            cur_beam=max(1, min(int(settings.beam_size), max_beam)),
        )

    backlog_hi: int = 12
    backlog_lo: int = 2
    latency_ratio_hi: float = 1.1
    latency_ratio_lo: float = 0.7

    cool_down_s: float = 2.0
    last_change_ts: float = 0.0

    def maybe_update(
        self, *, seg_qsize: int, last_latency_s: float, last_dur_s: float, now: float
    ) -> Tuple[int, Optional[str]]:
        if (now - float(self.last_change_ts)) < float(self.cool_down_s):
            return (int(self.cur_beam), None)

        dur = max(1e-6, float(last_dur_s))
        ratio = float(last_latency_s) / dur

        reason = None
        if seg_qsize >= int(self.backlog_hi) or ratio >= float(self.latency_ratio_hi):
            if self.cur_beam > self.min_beam:
                self.cur_beam -= 1
                self.last_change_ts = now
                reason = f"downshift (q={seg_qsize}, lat_ratio={ratio:.2f})"
        elif seg_qsize <= int(self.backlog_lo) and ratio <= float(self.latency_ratio_lo):
            if self.cur_beam < self.max_beam:
                self.cur_beam += 1
                self.last_change_ts = now
                reason = f"upshift (q={seg_qsize}, lat_ratio={ratio:.2f})"

        return (int(self.cur_beam), reason)
