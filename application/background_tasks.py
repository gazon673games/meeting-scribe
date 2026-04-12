from __future__ import annotations

from typing import Any, Callable, Optional, Protocol


class BackgroundTaskHandle(Protocol):
    def join(self, timeout: Optional[float] = None) -> None:
        ...


class BackgroundTaskRunner(Protocol):
    def start(
        self,
        *,
        name: str,
        target: Callable[..., None],
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> BackgroundTaskHandle:
        ...
