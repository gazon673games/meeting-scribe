from __future__ import annotations

import threading
from typing import Callable

from asr.application.ports import RealtimeWorkerRunnerPort, StopSignalPort, WorkerHandlePort


class ThreadRealtimeWorkerRunner(RealtimeWorkerRunnerPort):
    def create_stop_signal(self) -> StopSignalPort:
        return threading.Event()

    def start_worker(self, *, name: str, target: Callable[[], None]) -> WorkerHandlePort:
        worker = threading.Thread(target=target, name=str(name), daemon=True)
        worker.start()
        return worker
