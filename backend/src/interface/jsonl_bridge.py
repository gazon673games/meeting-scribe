from __future__ import annotations

import concurrent.futures
import json
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, TextIO


BackendHandler = Callable[[str, Dict[str, Any] | None], Any]


@dataclass
class JsonLineBridge:
    handler: BackendHandler
    stdin: TextIO = field(default_factory=lambda: sys.stdin)
    stdout: TextIO = field(default_factory=lambda: sys.stdout)
    stderr: TextIO = field(default_factory=lambda: sys.stderr)
    _write_lock: threading.RLock = field(default_factory=threading.RLock)

    def serve_forever(self) -> None:
        self.emit_event("backend_ready", {"ts": time.time()})
        with concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="bridge") as executor:
            for line in self.stdin:
                line = line.strip()
                if not line:
                    continue
                executor.submit(self._handle_line, line)

    def emit_event(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        self._write({"event": {"type": str(event_type), **dict(payload or {})}})

    def _handle_line(self, line: str) -> None:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")

            request_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {})
            if not isinstance(method, str) or not method:
                raise ValueError("request.method must be a non-empty string")
            if params is not None and not isinstance(params, dict):
                raise ValueError("request.params must be an object")

            result = self.handler(method, params)
            self._write({"id": request_id, "ok": True, "result": result})
        except Exception as exc:
            self._write(
                {
                    "id": _request_id_from_line(line),
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            traceback.print_exc(file=self.stderr)
            self.stderr.flush()

    def _write(self, message: Dict[str, Any]) -> None:
        with self._write_lock:
            self.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.stdout.flush()


def _request_id_from_line(line: str) -> Any:
    try:
        request = json.loads(line)
    except Exception:
        return None
    return request.get("id") if isinstance(request, dict) else None
