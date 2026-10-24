from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DiarSegment:
    t0: float
    t1: float
    speaker: str

    def overlap_duration(self, t0: float, t1: float) -> float:
        start = max(float(t0), float(self.t0))
        end = min(float(t1), float(self.t1))
        return max(0.0, end - start)


def pick_speaker(timeline: list[DiarSegment], t0: float, t1: float) -> str:
    best_label = 'S?'
    best_overlap_s = 0.0
    for segment in timeline:
        overlap_s = segment.overlap_duration(t0, t1)
        if overlap_s > best_overlap_s:
            best_overlap_s = overlap_s
            best_label = str(segment.speaker)
    return best_label
