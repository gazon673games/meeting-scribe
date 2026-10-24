from __future__ import annotations

import json
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any
from weakref import WeakValueDictionary

_logger = logging.getLogger(__name__)
_PATH_LOCKS: WeakValueDictionary[Path, threading.RLock] = WeakValueDictionary()
_PATH_LOCKS_GUARD = threading.Lock()


class JsonConfigRepository:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        key = self.path.resolve()
        with _PATH_LOCKS_GUARD:
            lock = _PATH_LOCKS.get(key)
            if lock is None:
                lock = threading.RLock()
                _PATH_LOCKS[key] = lock
            self._lock = lock

    def exists(self) -> bool:
        return self.path.exists()

    def read(self) -> dict[str, Any]:
        with self._lock:
            try:
                contents = self.path.read_text(encoding='utf-8')
            except FileNotFoundError:
                return {}
        raw = json.loads(contents)
        if not isinstance(raw, dict):
            _logger.warning(
                'Configuration root must be an object: %s', self.path
            )
            return {}
        return raw

    def write(self, config: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        contents = json.dumps(config, ensure_ascii=False, indent=2)
        with tempfile.TemporaryDirectory(
            prefix=f'.{self.path.name}.', dir=self.path.parent
        ) as directory:
            temporary_path = Path(directory) / self.path.name
            temporary_path.write_text(contents, encoding='utf-8')
            with self._lock:
                temporary_path.replace(self.path)
