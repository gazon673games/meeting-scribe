from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TextIO

from interface.backend_parts.session_dispatch import SERIAL_LIFECYCLE_METHODS
from interface.backend_protocol import BackendRequest, error_response

BackendHandler = Callable[[str, dict[str, Any] | None], Any]
_DEFAULT_MAX_WORKERS = 6


@dataclass
class JsonLineBridge:
    handler: BackendHandler
    stdin: TextIO = field(default_factory=lambda: sys.stdin)
    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)
    _write_lock: threading.RLock = field(default_factory=threading.RLock)
    max_pending_requests: int = 256
    _request_slots: threading.BoundedSemaphore = field(init=False, repr=False)
    on_disconnect: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_pending_requests, bool)
            or not isinstance(self.max_pending_requests, int)
            or self.max_pending_requests < 1
        ):
            raise ValueError('Request queue size must be a positive integer')
        self._request_slots = threading.BoundedSemaphore(
            self.max_pending_requests
        )

    def serve_forever(self) -> None:
        self.emit_event('backend_ready', {'ts': time.time()})
        max_workers = _bridge_max_workers()
        with (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix='bridge'
            ) as executor,
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='bridge-lifecycle'
            ) as lifecycle_executor,
        ):
            try:
                self._serve_requests(executor, lifecycle_executor)
            finally:
                if self.on_disconnect is not None:
                    self.on_disconnect()

    def _serve_requests(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        lifecycle_executor: concurrent.futures.ThreadPoolExecutor,
    ) -> None:
        for line in self.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except (ValueError, RecursionError) as error:
                self._write_error(None, error)
                continue
            method = (
                message.get('method') if isinstance(message, dict) else None
            )
            is_lifecycle = (
                isinstance(method, str) and method in SERIAL_LIFECYCLE_METHODS
            )
            target = lifecycle_executor if is_lifecycle else executor
            self._request_slots.acquire()
            try:
                future = target.submit(self._handle_request, message)
            except RuntimeError:
                self._request_slots.release()
                raise
            future.add_done_callback(self._complete_request)

    def _complete_request(
        self, future: concurrent.futures.Future[None]
    ) -> None:
        self._request_slots.release()
        error = future.exception()
        if error is not None:
            traceback.print_exception(
                type(error), error, error.__traceback__, file=self.stderr
            )
            self.stderr.flush()

    def emit_event(
        self, event_type: str, payload: dict[str, Any] | None = None
    ) -> None:
        self._write({'event': {'type': str(event_type), **dict(payload or {})}})

    def _handle_request(self, message: Any) -> None:
        request_id = message.get('id') if isinstance(message, dict) else None
        try:
            request = BackendRequest.from_message(message)
            result = self.handler(request.method, request.params)
            self._write(request.success(result))
        except Exception as error:
            self._write_error(request_id, error)

    def _write_error(self, request_id: Any, error: Exception) -> None:
        self._write(error_response(request_id, error))
        traceback.print_exception(
            type(error), error, error.__traceback__, file=self.stderr
        )
        self.stderr.flush()

    def _write(self, message: dict[str, Any]) -> None:
        with self._write_lock:
            self.stdout.write(
                json.dumps(message, ensure_ascii=False, separators=(',', ':'))
                + '\n'
            )
            self.stdout.flush()


def _bridge_max_workers() -> int:
    try:
        raw = int(
            str(os.environ.get('MEETING_SCRIBE_BRIDGE_WORKERS', '')).strip()
            or _DEFAULT_MAX_WORKERS
        )
    except ValueError:
        raw = _DEFAULT_MAX_WORKERS
    return max(2, min(16, raw))
