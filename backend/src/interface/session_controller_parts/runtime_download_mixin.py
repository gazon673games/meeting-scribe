from __future__ import annotations

from typing import Any, Dict


class RuntimeDownloadMixin:
    def begin_model_download(self, model_name: str) -> None:
        with self._lock:
            self._session_state.begin_model_download(model_name)
            self._model_download_info = {
                "model": model_name,
                "downloadedBytes": 0,
                "speedBps": 0.0,
                "message": "Starting download...",
            }
            self._emit("session_state_changed", {"state": "downloading_model", "model": model_name})

    def update_model_download_progress(self, info: Dict[str, Any]) -> None:
        with self._lock:
            self._model_download_info = {
                "model": self._model_download_info.get("model", ""),
                "downloadedBytes": int(info.get("downloadedBytes", 0)),
                "speedBps": float(info.get("speedBps", 0.0)),
                "message": str(info.get("message", "")),
            }

    def finish_model_download(self, error: str = "") -> None:
        with self._lock:
            model_name = self._model_download_info.get("model", "")
            self._session_state.finish_model_download(model_name=model_name, error=error)
            self._model_download_info = {}
            if error:
                self._last_error = error
                self._emit("session_error", {"message": error})
            self._emit("session_state_changed", {"state": "idle", "error": error})
