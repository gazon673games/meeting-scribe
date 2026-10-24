from __future__ import annotations

# ruff: noqa: F401
# This facade preserves imports used by existing callers.
from typing import Any

from asr.domain.profiles import (
    PROFILE_BALANCED,
    PROFILE_CUSTOM,
    PROFILE_QUALITY,
    PROFILE_REALTIME,
    PROFILE_ULTRA_FAST,
)
from asr.domain.profiles import (
    profile_defaults as _profile_defaults,
)
from asr.domain.profiles import (
    profile_requires_streaming as _profile_requires_streaming,
)


def profile_defaults(profile: str) -> dict[str, Any]:
    return _profile_defaults(profile)


def profile_requires_streaming(profile: str) -> bool:
    return _profile_requires_streaming(profile)
