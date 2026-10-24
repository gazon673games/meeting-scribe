from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


class BackgroundTaskHandle(Protocol):
    def join(self, timeout: float | None = None) -> None: ...


class BackgroundTaskRunner(Protocol):
    def start(
        self,
        *,
        name: str,
        target: Callable[..., None],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> BackgroundTaskHandle: ...
