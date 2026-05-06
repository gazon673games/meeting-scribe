from __future__ import annotations

from typing import Any, Dict

from asr.domain.profiles import (
    PROFILE_BALANCED,
    PROFILE_CUSTOM,
    PROFILE_QUALITY,
    PROFILE_REALTIME,
    PROFILE_ULTRA_FAST,
    profile_defaults as _profile_defaults,
    profile_requires_streaming as _profile_requires_streaming,
)


def profile_defaults(profile: str) -> Dict[str, Any]:
    return _profile_defaults(profile)


def profile_requires_streaming(profile: str) -> bool:
    return _profile_requires_streaming(profile)
