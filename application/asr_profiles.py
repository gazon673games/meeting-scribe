from __future__ import annotations

from typing import Any, Dict

from asr.domain.profiles import (
    PROFILE_BALANCED,
    PROFILE_CUSTOM,
    PROFILE_QUALITY,
    PROFILE_REALTIME,
    profile_defaults as _profile_defaults,
)


def profile_defaults(profile: str) -> Dict[str, Any]:
    return _profile_defaults(profile)
