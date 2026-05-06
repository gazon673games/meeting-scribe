from __future__ import annotations

from typing import Any, Dict, List, Optional

from application.model_policy import ASR_MODEL_NAMES
from interface.backend_parts.system_utils import int_or_zero, scan_compatible_asr_models


def _ui_model(config: Dict[str, Any]) -> str:
    ui = config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}
    return str(ui.get("model", "") or "").strip()


class BackendAsrModelsMixin:
    def list_models(self, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        config = self._read_config()
        models_dir = self._models_dir_from_params(params, config)
        downloads = self._download_snapshot()
        current_model = _ui_model(config)
        cache_key = ("asr", str(models_dir), current_model)

        if not self._active_download_count(downloads):
            hit = self._catalog_cache_get(cache_key)
            if hit is not None:
                return hit

        models = self._build_model_list(ASR_MODEL_NAMES, models_dir, downloads, current_model)
        result = {"models": models, "modelsDir": str(models_dir), "activeDownloads": self._active_download_count(downloads)}
        if not result["activeDownloads"]:
            self._catalog_cache_put(cache_key, result)
        return result

    def _build_model_list(
        self,
        recommended_names: List[str],
        models_dir: Any,
        downloads: Dict,
        current_model: str,
    ) -> List[Dict[str, Any]]:
        from application.model_download import normalize_model_reference

        recommended_refs = {normalize_model_reference(n) for n in recommended_names}
        models: List[Dict[str, Any]] = [
            self._recommended_model_record(n, models_dir, downloads)
            for n in recommended_names
        ]
        models += self._local_model_records(models_dir, downloads, recommended_refs)
        self._add_unlisted_models(models, downloads, current_model, models_dir)
        return models

    def _local_model_records(self, models_dir: Any, downloads: Dict, skip_refs: set) -> List[Dict[str, Any]]:
        from application.model_download import normalize_model_reference, scan_local_models

        result = []
        for rec in scan_local_models(models_dir):
            name = str(rec.get("name") or "")
            if normalize_model_reference(name) not in skip_refs:
                result.append({**rec, **self._download_fields(name, downloads)})
        return result

    def _add_unlisted_models(self, models: List, downloads: Dict, current_model: str, models_dir: Any) -> None:
        from application.model_download import normalize_model_reference

        known = {normalize_model_reference(str(m.get("name") or "")) for m in models}
        if current_model:
            known = self._add_if_missing(models, known, current_model, models_dir, downloads)
        for name in downloads:
            known = self._add_if_missing(models, known, name, models_dir, downloads)

    def _add_if_missing(self, models: List, known: set, name: str, models_dir: Any, downloads: Dict) -> set:
        from application.model_download import normalize_model_reference

        ref = normalize_model_reference(name)
        if ref and ref not in known:
            models.append(self._custom_model_record(name, models_dir, downloads))
            known.add(ref)
        return known

    def _recommended_model_record(self, name: str, models_dir: Any, downloads: Dict) -> Dict[str, Any]:
        from application.model_download import is_builtin_model, is_model_cached

        cached = is_model_cached(name, models_dir=models_dir)
        return {
            "name": name,
            "label": name,
            "cached": cached,
            "compatible": cached,
            "status": "compatible" if cached else "recommended",
            "source": "recommended",
            "builtin": is_builtin_model(name),
            "recommended": True,
            "downloadable": True,
            "deletable": False,
            **self._download_fields(name, downloads),
        }

    def _custom_model_record(self, name: str, models_dir: Any, downloads: Dict) -> Dict[str, Any]:
        from application.model_download import is_model_cached

        cached = is_model_cached(name, models_dir=models_dir)
        return {
            "name": name,
            "label": name,
            "cached": cached,
            "compatible": cached,
            "status": "compatible" if cached else "unknown_remote",
            "source": "custom",
            "builtin": False,
            "recommended": False,
            "downloadable": True,
            "deletable": False,
            **self._download_fields(name, downloads),
        }

    def download_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from application.model_download import download_model_async, normalize_model_reference

        name = str(params.get("name") or "").strip()
        if not name:
            raise ValueError("download_model requires params.name")
        name = normalize_model_reference(name)
        models_dir = self._models_dir_from_params(params)
        if self._download_record(name).get("state") == "downloading":
            return {"started": False, "message": "Already downloading"}
        if "useProxy" in params or "use_proxy" in params:
            use_proxy = bool(params.get("useProxy", params.get("use_proxy", False)))
            proxy = str(params.get("proxy") or "").strip() if use_proxy else ""
        else:
            proxy = self._model_download_proxy()
        self._set_download_state(
            name,
            {
                "state": "downloading",
                "message": "Starting...",
                "error": "",
                "downloadedBytes": 0,
                "speedBps": 0,
                "proxy": bool(proxy),
            },
        )

        def on_progress(update: Any) -> None:
            payload = update if isinstance(update, dict) else {"message": str(update)}
            self._set_download_state(
                name,
                {
                    "state": "downloading",
                    "message": str(payload.get("message") or "Downloading..."),
                    "error": "",
                    "downloadedBytes": int_or_zero(payload.get("downloadedBytes")),
                    "speedBps": float(payload.get("speedBps") or 0.0),
                    "proxy": bool(proxy),
                },
            )

        def on_done(error: Optional[str]) -> None:
            if error:
                self._set_download_state(
                    name,
                    {
                        "state": "error",
                        "message": "",
                        "error": error,
                        "downloadedBytes": self._download_record(name).get("downloadedBytes", 0),
                        "speedBps": 0,
                        "proxy": bool(proxy),
                    },
                )
            else:
                self._set_download_state(
                    name,
                    {
                        "state": "done",
                        "message": "Downloaded",
                        "error": "",
                        "downloadedBytes": self._download_record(name).get("downloadedBytes", 0),
                        "speedBps": 0,
                        "proxy": bool(proxy),
                    },
                )

        download_model_async(name, on_progress, on_done, models_dir=models_dir, proxy=proxy)
        return {"started": True, "message": f"Downloading {name}...", "proxy": bool(proxy)}

    def delete_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from application.model_download import delete_local_model

        name = str(params.get("name") or "").strip()
        if self._download_record(name).get("state") == "downloading":
            raise RuntimeError("Wait until the model download finishes before deleting it")
        if self._model_is_selected_or_running(name):
            raise RuntimeError("Cannot delete the model that is selected or currently used by ASR")
        delete_local_model(name, models_dir=self._models_dir_from_params(params))
        self._catalog_cache_clear()
        return {"deleted": True, "name": name}

    def model_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from application.model_download import model_metadata

        name = str(params.get("name") or "").strip()
        if not name:
            raise ValueError("model_metadata requires params.name")
        return model_metadata(name, models_dir=self._models_dir_from_params(params))

    def _asr_model_options(self, config: Dict[str, Any]) -> List[str]:
        local = scan_compatible_asr_models(self._models_dir(config))
        ui = config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}
        current = str(ui.get("model", "") or "").strip()
        seen: set[str] = set()
        out: List[str] = []
        for name in [*ASR_MODEL_NAMES, *local, current]:
            text = str(name or "").strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out

