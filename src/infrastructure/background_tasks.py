from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from application.background_tasks import BackgroundTaskHandle, BackgroundTaskRunner


class ThreadBackgroundTaskRunner(BackgroundTaskRunner):
    def start(
        self,
        *,
        name: str,
        target: Callable[..., None],
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None,
    ) -> BackgroundTaskHandle:
        worker = threading.Thread(
            target=target,
            args=tuple(args),
            kwargs=dict(kwargs or {}),
            name=str(name),
            daemon=True,
        )
        worker.start()
        return worker
