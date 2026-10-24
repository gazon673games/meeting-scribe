from __future__ import annotations

from typing import Any


def first_param(
    params: dict[str, Any], *keys: str, default: object = None
) -> object:
    for key in keys:
        if key in params:
            return params[key]
    return default
