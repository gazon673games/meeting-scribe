from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from interface.session_controller import HeadlessSessionController


class BackendBaseMixin:
    def _read_config(self) -> Dict[str, Any]:
        try:
            config = self.config_repository.read()
        except Exception:
            return {}
        return config if isinstance(config, dict) else {}

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        sink = self._event_sink
        if sink is None:
            return
        try:
            sink(event_type, {"ts": time.time(), **payload})
        except Exception:
            pass

    def _models_dir(self, config: Dict[str, Any] | None = None) -> Path:
        cfg = config if isinstance(config, dict) else self._read_config()
        models = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
        raw = str(models.get("cache_dir", "") or "").strip()
        return Path(raw).expanduser().resolve() if raw else Path(self.project_root).resolve() / "models"

    def _models_dir_from_params(self, params: Dict[str, Any] | None = None, config: Dict[str, Any] | None = None) -> Path:
        raw = ""
        if isinstance(params, dict):
            raw = str(params.get("modelsDir", params.get("models_dir", "")) or "").strip()
        return Path(raw).expanduser().resolve() if raw else self._models_dir(config)

    def _llm_models_dir_from_params(self, params: Dict[str, Any] | None = None) -> Path | None:
        raw = ""
        if isinstance(params, dict):
            raw = str(params.get("modelsDir", params.get("models_dir", "")) or "").strip()
        return Path(raw).expanduser().resolve() if raw else None

    def _require_session_controller(self) -> HeadlessSessionController:
        if self.session_controller is None:
            raise RuntimeError("Session controller is not configured")
        return self.session_controller

