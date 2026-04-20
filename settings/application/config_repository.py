from __future__ import annotations

from typing import Any, Dict, Protocol


class ConfigRepository(Protocol):
    def exists(self) -> bool:
        ...

    def read(self) -> Dict[str, Any]:
        ...

    def write(self, config: Dict[str, Any]) -> None:
        ...
