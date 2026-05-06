from __future__ import annotations

from typing import Any, Dict, Optional

from interface.backend_parts.system_utils import int_or_zero


class BackendExtraModelsMixin:
    def list_diarization_models(self, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        from application.diarization_model_download import list_diarization_models

        models_dir = self._models_dir_from_params(params)
        downloads = self._diarization_download_snapshot()
        cache_key = ("diar", str(models_dir))
        if self._active_download_count(downloads) <= 0:
            cached = self._catalog_cache_get(cache_key)
            if cached is not None:
                return cached
        result = list_diarization_models(
            project_root=self.project_root,
            models_dir=models_dir,
            downloads=downloads,
        )
        if self._active_download_count(downloads) <= 0:
            self._catalog_cache_put(cache_key, result)
        return result

    def download_diarization_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from application.diarization_model_download import download_diarization_model_async

        name = str(params.get("name") or "").strip()
        if not name:
            raise ValueError("download_diarization_model requires params.name")
        if self._diarization_download_record(name).get("state") == "downloading":
            return {"started": False, "message": "Already downloading"}
        if "useProxy" in params or "use_proxy" in params:
            use_proxy = bool(params.get("useProxy", params.get("use_proxy", False)))
            proxy = str(params.get("proxy") or "").strip() if use_proxy else ""
        else:
            proxy = self._model_download_proxy()
        self._set_diarization_download_state(
            name,
            {
                "state": "downloading",
                "message": "Starting...",
                "error": "",
                "downloadedBytes": 0,
                "totalBytes": 0,
                "speedBps": 0,
                "proxy": bool(proxy),
            },
        )

        def on_progress(update: Any) -> None:
            payload = update if isinstance(update, dict) else {"message": str(update)}
            self._set_diarization_download_state(
                name,
                {
                    "state": "downloading",
                    "message": str(payload.get("message") or "Downloading..."),
                    "error": "",
                    "downloadedBytes": int_or_zero(payload.get("downloadedBytes")),
                    "totalBytes": int_or_zero(payload.get("totalBytes")),
                    "speedBps": float(payload.get("speedBps") or 0.0),
                    "path": str(payload.get("path") or ""),
                    "proxy": bool(proxy),
                },
            )

        def on_done(error: Optional[str]) -> None:
            previous = self._diarization_download_record(name)
            self._set_diarization_download_state(
                name,
                {
                    "state": "error" if error else "done",
                    "message": "" if error else "Downloaded",
                    "error": str(error or ""),
                    "downloadedBytes": int_or_zero(previous.get("downloadedBytes")),
                    "totalBytes": int_or_zero(previous.get("totalBytes")),
                    "speedBps": 0,
                    "path": str(previous.get("path") or ""),
                    "proxy": bool(proxy),
                },
            )

        download_diarization_model_async(
            name=name,
            project_root=self.project_root,
            models_dir=self._models_dir_from_params(params),
            proxy=proxy,
            on_progress=on_progress,
            on_done=on_done,
        )
        return {"started": True, "message": f"Downloading Speaker ID model {name}...", "proxy": bool(proxy)}

    def delete_diarization_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from application.diarization_model_download import delete_diarization_model

        path = str(params.get("path") or "").strip()
        if not path:
            raise ValueError("delete_diarization_model requires params.path")
        if self._diarization_model_is_selected_or_running(path):
            raise RuntimeError("Cannot delete the Speaker ID model that is selected or currently used by ASR")
        delete_diarization_model(
            project_root=self.project_root,
            path=path,
            models_dir=self._models_dir_from_params(params),
        )
        self._catalog_cache_clear()
        return {"deleted": True, "path": path}

    def list_llm_models(self, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        from application.llm_model_download import list_llm_models

        models_dir = self._llm_models_dir_from_params(params)
        downloads = self._llm_download_snapshot()
        cache_key = ("llm", str(models_dir))
        if self._active_download_count(downloads) <= 0:
            cached = self._catalog_cache_get(cache_key)
            if cached is not None:
                return cached
        result = list_llm_models(
            project_root=self.project_root,
            models_dir=models_dir,
            downloads=downloads,
        )
        if self._active_download_count(downloads) <= 0:
            self._catalog_cache_put(cache_key, result)
        return result

    def download_llm_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from application.llm_model_download import download_llm_model_async, parse_llm_source

        name = str(params.get("name") or "").strip()
        if not name:
            raise ValueError("download_llm_model requires params.name")
        source = parse_llm_source(name)
        key = source.filename or source.folder or source.repo_id or name
        if self._llm_download_record(key).get("state") == "downloading":
            return {"started": False, "message": "Already downloading"}
        if "useProxy" in params or "use_proxy" in params:
            use_proxy = bool(params.get("useProxy", params.get("use_proxy", False)))
            proxy = str(params.get("proxy") or "").strip() if use_proxy else ""
        else:
            proxy = self._model_download_proxy()
        self._set_llm_download_state(
            key,
            {
                "state": "downloading",
                "message": "Starting...",
                "error": "",
                "downloadedBytes": 0,
                "totalBytes": 0,
                "speedBps": 0,
                "path": "",
                "proxy": bool(proxy),
            },
        )

        def on_progress(update: Any) -> None:
            payload = update if isinstance(update, dict) else {"message": str(update)}
            self._set_llm_download_state(
                key,
                {
                    "state": "downloading",
                    "message": str(payload.get("message") or "Downloading..."),
                    "error": "",
                    "downloadedBytes": int_or_zero(payload.get("downloadedBytes")),
                    "totalBytes": int_or_zero(payload.get("totalBytes")),
                    "speedBps": float(payload.get("speedBps") or 0.0),
                    "path": str(payload.get("path") or ""),
                    "proxy": bool(proxy),
                },
            )

        def on_done(error: Optional[str]) -> None:
            previous = self._llm_download_record(key)
            self._set_llm_download_state(
                key,
                {
                    "state": "error" if error else "done",
                    "message": "" if error else "Downloaded",
                    "error": str(error or ""),
                    "downloadedBytes": int_or_zero(previous.get("downloadedBytes")),
                    "totalBytes": int_or_zero(previous.get("totalBytes")),
                    "speedBps": 0,
                    "path": str(previous.get("path") or ""),
                    "proxy": bool(proxy),
                },
            )

        download_llm_model_async(
            name=name,
            project_root=self.project_root,
            models_dir=self._llm_models_dir_from_params(params),
            proxy=proxy,
            on_progress=on_progress,
            on_done=on_done,
        )
        label = source.filename or source.repo_id or name
        return {"started": True, "message": f"Downloading language model {label}...", "proxy": bool(proxy)}

    def delete_llm_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from application.llm_model_download import delete_llm_model

        path = str(params.get("path") or "").strip()
        if not path:
            raise ValueError("delete_llm_model requires params.path")
        delete_llm_model(project_root=self.project_root, path=path, models_dir=self._llm_models_dir_from_params(params))
        self._catalog_cache_clear()
        return {"deleted": True, "path": path}

