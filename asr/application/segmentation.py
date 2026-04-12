from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SegmenterConfig:
    vad_energy_threshold: float
    vad_hangover_ms: int
    vad_min_speech_ms: int
    vad_band_ratio_min: float
    vad_voiced_min: float
    vad_pre_speech_ms: int
    vad_min_end_silence_ms: int
    min_segment_ms: int
    agc_enabled: bool
    agc_target_rms: float
    agc_max_gain: float
    agc_alpha: float


class AudioSegmenterPort(Protocol):
    def reset_runtime(self) -> None:
        ...

    def feed_packet(self, *, mode: str, pkt: dict) -> None:
        ...
