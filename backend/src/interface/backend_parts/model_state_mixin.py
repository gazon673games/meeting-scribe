from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from interface.backend_parts.system_utils import int_or_zero


class BackendModelStateMixin:
    def _download_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._download_lock:
            return {str(name): dict(record) for name, record in self._downloads.items()}

    def _diarization_download_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._diarization_download_lock:
            return {str(name): dict(record) for name, record in self._diarization_downloads.items()}

    def _llm_download_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._llm_download_lock:
            return {str(name): dict(record) for name, record in self._llm_downloads.items()}

    def _download_record(self, name: str) -> Dict[str, Any]:
        from application.model_download import normalize_model_reference

        key = normalize_model_reference(name)
        with self._download_lock:
            return dict(self._downloads.get(name) or self._downloads.get(key) or {})

    def _diarization_download_record(self, name: str) -> Dict[str, Any]:
        with self._diarization_download_lock:
            return dict(self._diarization_downloads.get(str(name)) or {})

    def _llm_download_record(self, name: str) -> Dict[str, Any]:
        with self._llm_download_lock:
            return dict(self._llm_downloads.get(str(name)) or {})

    def _download_fields(self, name: str, downloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        from application.model_download import normalize_model_reference

        key = normalize_model_reference(name)
        dl = downloads.get(name) or downloads.get(key) or {}
        return {
            "downloading": dl.get("state") == "downloading",
            "downloadDone": dl.get("state") == "done",
            "downloadError": str(dl.get("error") or ""),
            "downloadMessage": str(dl.get("message") or ""),
            "downloadedBytes": int_or_zero(dl.get("downloadedBytes")),
            "speedBps": float(dl.get("speedBps") or 0.0),
            "downloadUsesProxy": bool(dl.get("proxy", False)),
        }

    def _catalog_cache_get(self, key: Tuple[Any, ...], *, max_age_s: float = 10.0) -> Optional[Dict[str, Any]]:
        now = time.monotonic()
        with self._catalog_cache_lock:
            cached = self._catalog_cache.get(tuple(key))
            if cached is None:
                return None
            ts, value = cached
            if now - float(ts) > float(max_age_s):
                self._catalog_cache.pop(tuple(key), None)
                return None
            return copy.deepcopy(value)

    def _catalog_cache_put(self, key: Tuple[Any, ...], value: Dict[str, Any]) -> None:
        with self._catalog_cache_lock:
            self._catalog_cache[tuple(key)] = (time.monotonic(), copy.deepcopy(value))

    def _catalog_cache_clear(self) -> None:
        with self._catalog_cache_lock:
            self._catalog_cache.clear()

    def _set_download_state(self, name: str, record: Dict[str, Any]) -> None:
        with self._download_lock:
            self._downloads[str(name)] = dict(record)
            active_downloads = self._active_download_count(self._downloads)
        self._catalog_cache_clear()
        self._emit(
            "model_download_updated",
            {
                "model": str(name),
                **dict(record),
                "activeDownloads": active_downloads,
            },
        )

    def _set_diarization_download_state(self, name: str, record: Dict[str, Any]) -> None:
        with self._diarization_download_lock:
            self._diarization_downloads[str(name)] = dict(record)
            active_downloads = self._active_download_count(self._diarization_downloads)
        self._catalog_cache_clear()
        self._emit(
            "diarization_model_download_updated",
            {
                "model": str(name),
                **dict(record),
                "activeDownloads": active_downloads,
            },
        )

    def _set_llm_download_state(self, name: str, record: Dict[str, Any]) -> None:
        with self._llm_download_lock:
            self._llm_downloads[str(name)] = dict(record)
            active_downloads = self._active_download_count(self._llm_downloads)
        self._catalog_cache_clear()
        self._emit(
            "llm_model_download_updated",
            {
                "model": str(name),
                **dict(record),
                "activeDownloads": active_downloads,
            },
        )

    def _active_download_count(self, downloads: Dict[str, Dict[str, Any]] | None = None) -> int:
        source = downloads
        if source is None:
            with self._download_lock:
                source = dict(self._downloads)
        return sum(1 for record in source.values() if record.get("state") == "downloading")

    def _model_download_proxy(self, config: Dict[str, Any] | None = None) -> str:
        cfg = config if isinstance(config, dict) else self._read_config()
        models = cfg.get("models", {}) if isinstance(cfg.get("models"), dict) else {}
        if not bool(models.get("use_proxy", False)):
            return ""
        model_proxy = str(models.get("proxy") or "").strip()
        if model_proxy:
            return model_proxy
        codex = cfg.get("codex", {}) if isinstance(cfg.get("codex"), dict) else {}
        return str(codex.get("proxy") or "").strip()

    def _model_is_selected_or_running(self, name: str) -> bool:
        from application.model_download import normalize_model_reference

        wanted = normalize_model_reference(name)
        config = self._read_config()
        current = str((config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}).get("model", "") or "").strip()
        if wanted and normalize_model_reference(current) == wanted:
            return True
        session = self._session_snapshot()
        if session.get("running") and wanted and normalize_model_reference(current) == wanted:
            return True
        return False

    def _diarization_model_is_selected_or_running(self, path: str) -> bool:
        wanted = str(Path(path).expanduser().resolve())
        config = self._read_config()
        asr = config.get("asr", {}) if isinstance(config.get("asr"), dict) else {}
        current = str(asr.get("diar_sherpa_embedding_model_path") or "").strip()
        if current and str(Path(current).expanduser().resolve()) == wanted:
            return True
        session = self._session_snapshot()
        return bool(session.get("running") and current and str(Path(current).expanduser().resolve()) == wanted)

