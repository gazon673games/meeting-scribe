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

    def to_event_dict(self) -> dict:
        return {
            "energy_threshold": self.vad_energy_threshold,
            "hangover_ms": self.vad_hangover_ms,
            "min_speech_ms": self.vad_min_speech_ms,
            "band_ratio_min": self.vad_band_ratio_min,
            "voiced_min": self.vad_voiced_min,
            "pre_speech_ms": self.vad_pre_speech_ms,
            "min_end_silence_ms": self.vad_min_end_silence_ms,
            "min_segment_ms": self.min_segment_ms,
        }


class AudioSegmenterPort(Protocol):
    def reset_runtime(self) -> None:
        ...

    def feed_packet(self, *, mode: str, pkt: dict) -> None:
        ...
