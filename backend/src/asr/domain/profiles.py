from __future__ import annotations

from typing import Any, Dict

PROFILE_REALTIME = "Realtime"
PROFILE_BALANCED = "Balanced"
PROFILE_QUALITY = "Quality"
PROFILE_CUSTOM = "Custom"


def profile_defaults(profile: str) -> Dict[str, Any]:
    p = (profile or "").strip().lower()
    if p == PROFILE_REALTIME.lower():
        return {
            "compute_type": "int8_float16",
            "beam_size": 2,
            "endpoint_silence_ms": 450.0,
            "max_segment_s": 5.0,
            "overlap_ms": 120.0,
            "vad_energy_threshold": 0.0055,
            "overload_strategy": "drop_old",
            "overload_enter_qsize": 14,
            "overload_exit_qsize": 5,
            "overload_hard_qsize": 22,
            "overload_beam_cap": 1,
            "overload_max_segment_s": 3.5,
            "overload_overlap_ms": 80.0,
        }

    if p == PROFILE_QUALITY.lower():
        return {
            "compute_type": "float16",
            "beam_size": 6,
            "endpoint_silence_ms": 900.0,
            "max_segment_s": 12.0,
            "overlap_ms": 320.0,
            "vad_energy_threshold": 0.0052,
            "overload_strategy": "keep_all",
            "overload_enter_qsize": 22,
            "overload_exit_qsize": 8,
            "overload_hard_qsize": 40,
            "overload_beam_cap": 3,
            "overload_max_segment_s": 6.0,
            "overload_overlap_ms": 160.0,
        }

    return {
        "compute_type": "float16",
        "beam_size": 5,
        "endpoint_silence_ms": 650.0,
        "max_segment_s": 7.0,
        "overlap_ms": 200.0,
        "vad_energy_threshold": 0.0055,
        "overload_strategy": "drop_old",
        "overload_enter_qsize": 18,
        "overload_exit_qsize": 6,
        "overload_hard_qsize": 28,
        "overload_beam_cap": 2,
        "overload_max_segment_s": 5.0,
        "overload_overlap_ms": 120.0,
    }
