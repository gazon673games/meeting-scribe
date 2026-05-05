from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StreamingSegmenterConfig:
    chunk_interval_s: float = 1.0
    endpoint_silence_ms: float = 300.0
    max_segment_s: float = 30.0
    vad_energy_threshold: float = 0.02
    vad_hangover_ms: float = 400.0
    vad_min_speech_ms: float = 100.0
    vad_band_ratio_min: float = 0.15
    vad_voiced_min: float = 0.1
    vad_pre_speech_ms: float = 120.0
    vad_min_end_silence_ms: float = 0.0
    agc_enabled: bool = True
    agc_target_rms: float = 0.05
    agc_max_gain: float = 10.0
    agc_alpha: float = 0.95


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
