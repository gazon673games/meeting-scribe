from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from application.background_tasks import (
    BackgroundTaskHandle,
    BackgroundTaskRunner,
)


@dataclass
class ThreadBackgroundTaskRunner(BackgroundTaskRunner):
    _workers: set[threading.Thread] = field(default_factory=set, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _closing: bool = field(default=False, init=False)

    def start(
        self,
        *,
        name: str,
        target: Callable[..., None],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> BackgroundTaskHandle:
        arguments = tuple(args)
        keyword_arguments = dict(kwargs or {})

        def run() -> None:
            try:
                target(*arguments, **keyword_arguments)
            finally:
                with self._lock:
                    self._workers.discard(worker)

        with self._lock:
            if self._closing:
                raise RuntimeError('Background task runner is shutting down')
            worker = threading.Thread(target=run, name=str(name), daemon=True)
            self._workers.add(worker)
            try:
                worker.start()
            except BaseException:
                self._workers.discard(worker)
                raise
        return worker

    def shutdown(self, timeout_s: float = 5.0) -> None:
        with self._lock:
            self._closing = True
            workers = list(self._workers)
        deadline = time.monotonic() + max(0.0, timeout_s)
        for worker in workers:
            if worker is threading.current_thread():
                continue
            worker.join(max(0.0, deadline - time.monotonic()))
        live = [
            worker.name
            for worker in workers
            if worker is not threading.current_thread() and worker.is_alive()
        ]
        if live:
            raise TimeoutError(
                f'Background tasks did not stop: {", ".join(live)}'
            )


model_download_tasks = ThreadBackgroundTaskRunner()
