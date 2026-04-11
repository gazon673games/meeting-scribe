from __future__ import annotations

from typing import List, Tuple

from audio.devices import (
    LOOPBACK_SOURCE_TYPE,
    MIC_SOURCE_TYPE,
    list_input_devices as _list_input_devices,
    list_loopback_devices as _list_loopback_devices,
)


def list_loopback_devices() -> List[Tuple[str, object]]:
    return _list_loopback_devices()


def list_input_devices() -> List[Tuple[str, int]]:
    return _list_input_devices()
