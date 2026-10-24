from __future__ import annotations

from application.device_catalog import DeviceCatalog
from audio.infrastructure.devices import (
    list_input_devices as _list_input_devices,
)
from audio.infrastructure.devices import (
    list_loopback_devices as _list_loopback_devices,
)


class SoundDeviceCatalog(DeviceCatalog):
    def list_loopback_devices(self) -> list[tuple[str, object]]:
        return _list_loopback_devices()

    def list_input_devices(self) -> list[tuple[str, int]]:
        return _list_input_devices()
