from __future__ import annotations

from typing import List, Tuple

from application.device_catalog import DeviceCatalog
from audio.devices import (
    list_input_devices as _list_input_devices,
    list_loopback_devices as _list_loopback_devices,
)


class SoundDeviceCatalog(DeviceCatalog):
    def list_loopback_devices(self) -> List[Tuple[str, object]]:
        return _list_loopback_devices()

    def list_input_devices(self) -> List[Tuple[str, int]]:
        return _list_input_devices()
