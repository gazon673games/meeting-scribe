from __future__ import annotations

from typing import Any, Protocol


class ConfigRepository(Protocol):
    def exists(self) -> bool: ...

    def read(self) -> dict[str, Any]: ...

    def write(self, config: dict[str, Any]) -> None: ...
