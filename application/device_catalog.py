from __future__ import annotations

from typing import List, Protocol, Tuple

LOOPBACK_SOURCE_TYPE = "System audio (WASAPI loopback)"
MIC_SOURCE_TYPE = "Microphone (input device)"


class DeviceCatalog(Protocol):
    def list_loopback_devices(self) -> List[Tuple[str, object]]:
        ...

    def list_input_devices(self) -> List[Tuple[str, int]]:
        ...
