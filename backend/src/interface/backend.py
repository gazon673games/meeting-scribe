from __future__ import annotations

import ctypes
import copy
import importlib.util
import os
import platform
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from application.asr_language import SUPPORTED_ASR_LANGUAGES
from application.asr_profiles import (
    PROFILE_BALANCED,
    PROFILE_CUSTOM,
    PROFILE_QUALITY,
    PROFILE_REALTIME,
    PROFILE_ULTRA_FAST,
    profile_defaults,
    profile_requires_streaming,
)
from application.device_catalog import DeviceCatalog
from application.model_policy import ASR_MODEL_NAMES
from interface.assistant_controller import AssistantController
from interface.session_controller import HeadlessSessionController
from settings.application.config_repository import ConfigRepository


DeviceToken = Tuple[str, object, str]


def _ui_model(config: Dict[str, Any]) -> str:
    ui = config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}
    return str(ui.get("model", "") or "").strip()


@dataclass
class ElectronBackend:
    project_root: Path
    config_repository: ConfigRepository
    device_catalog: DeviceCatalog
    session_controller: HeadlessSessionController | None = None
    assistant_controller: AssistantController | None = None
    _device_tokens: Dict[str, DeviceToken] = field(default_factory=dict)
    _hardware_cache: Dict[str, Any] | None = field(default=None, init=False, repr=False)
    _hardware_cache_ts: float = field(default=0.0, init=False, repr=False)
    _gpu_cache: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _gpu_cache_ts: float = field(default=0.0, init=False, repr=False)
    _resource_cpu_time_s: float = field(default=0.0, init=False, repr=False)
    _resource_wall_time_s: float = field(default=0.0, init=False, repr=False)
    _resource_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _catalog_cache: Dict[Tuple[Any, ...], Tuple[float, Dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _catalog_cache_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _device_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _downloads: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _download_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _diarization_downloads: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _diarization_download_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _llm_downloads: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _llm_download_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _event_sink: Any = field(default=None, init=False, repr=False)

    protocol_version: int = 1

    def set_event_sink(self, event_sink) -> None:  # noqa: ANN001
        self._event_sink = event_sink
        if self.session_controller is not None:
            self.session_controller.set_event_sink(event_sink)
        if self.assistant_controller is not None:
            self.assistant_controller.set_event_sink(event_sink)

    def ping(self, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {"pong": True, "echo": dict(params or {})}

    def get_state(self) -> Dict[str, Any]:
        config = self._read_config()
        ui = config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}
        asr = config.get("asr", {}) if isinstance(config.get("asr"), dict) else {}
        codex = config.get("codex", {}) if isinstance(config.get("codex"), dict) else {}
        profiles = codex.get("profiles", []) if isinstance(codex, dict) else []
        session = self._session_snapshot()
        assistant = self._assistant_snapshot()

        return {
            "protocolVersion": self.protocol_version,
            "appName": "Meeting Scribe",
            "session": session,
            "assistant": assistant,
            "paths": {
                "projectRoot": str(self.project_root),
                "config": str(Path(self.config_repository.path)),  # type: ignore[attr-defined]
                "models": str(self._models_dir()),
            },
            "hardware": self._hardware_snapshot(),
            "configSummary": {
                "language": str(ui.get("lang", "")),
                "model": str(ui.get("model", "")),
                "profile": str(ui.get("profile", "")),
                "asrEnabled": bool(ui.get("asr_enabled", False)),
                "asrMode": "split" if int(ui.get("asr_mode", 0) or 0) == 1 else "mix",
                "wavEnabled": bool(ui.get("wav_enabled", False)),
                "offlineOnStop": bool(ui.get("offline_on_stop", False)),
                "device": str(asr.get("device", "")),
                "computeType": str(asr.get("compute_type", "")),
                "cpuThreads": _int_or_default(asr.get("cpu_threads", 0), 0),
                "numWorkers": _int_or_default(asr.get("num_workers", 1), 1),
                "diarizationEnabled": bool(asr.get("diarization_enabled", False)),
                "diarBackend": str(asr.get("diar_backend", "online") or "online"),
                "codexEnabled": bool(codex.get("enabled", False)) if isinstance(codex, dict) else False,
                "codexProfiles": len(profiles) if isinstance(profiles, list) else 0,
            },
            "options": {
                "languages": list(SUPPORTED_ASR_LANGUAGES),
                "asrProfiles": [PROFILE_ULTRA_FAST, PROFILE_REALTIME, PROFILE_QUALITY, PROFILE_CUSTOM],
                "asrModels": self._asr_model_options(config),
                "asrModes": [
                    {"id": "mix", "label": "MIX (master)"},
                    {"id": "split", "label": "SPLIT (all sources)"},
                ],
                "asrDevices": ["cuda", "cpu"],
                "computeTypes": ["int8_float16", "float16", "int8", "int8_float32", "float32"],
                "diarizationBackends": ["online", "sherpa_onnx", "nemo", "pyannote"],
                "diarizationProviders": ["cpu", "cuda"],
                "overloadStrategies": ["drop_old", "keep_all"],
                "profileDefaults": {
                    PROFILE_ULTRA_FAST: profile_defaults(PROFILE_ULTRA_FAST),
                    PROFILE_REALTIME: profile_defaults(PROFILE_REALTIME),
                    PROFILE_BALANCED: profile_defaults(PROFILE_BALANCED),
                    PROFILE_QUALITY: profile_defaults(PROFILE_QUALITY),
                    PROFILE_CUSTOM: profile_defaults(PROFILE_BALANCED),
                },
                "streamingLockedProfiles": [
                    profile for profile in [PROFILE_ULTRA_FAST, PROFILE_REALTIME, PROFILE_QUALITY]
                    if profile_requires_streaming(profile)
                ],
            },
            "capabilities": {
                "config": True,
                "devices": True,
                "sourceControl": self.session_controller is not None,
                "sessionControl": self.session_controller is not None,
                "assistant": self.assistant_controller is not None,
                "liveTranscript": self.session_controller is not None,
                "perProcessAudio": _per_process_audio_supported(),
            },
        }

    def get_runtime_state(self) -> Dict[str, Any]:
        return {
            "protocolVersion": self.protocol_version,
            "session": self._session_snapshot(),
        }

    def get_config(self) -> Dict[str, Any]:
        return self._read_config()

    def get_resource_usage(self, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        params = params if isinstance(params, dict) else {}
        include_gpu = bool(params.get("includeGpu", params.get("include_gpu", True)))
        with self._resource_lock:
            now = time.monotonic()
            cpu_time_s = time.process_time()
            cpu_pct = 0.0
            if self._resource_wall_time_s > 0.0:
                wall_delta = max(1e-6, now - self._resource_wall_time_s)
                cpu_delta = max(0.0, cpu_time_s - self._resource_cpu_time_s)
                cpu_pct = min(100.0, (cpu_delta / wall_delta / max(1, os.cpu_count() or 1)) * 100.0)
            self._resource_wall_time_s = now
            self._resource_cpu_time_s = cpu_time_s
            return {
                "pid": os.getpid(),
                "cpuPct": cpu_pct,
                "memoryBytes": _current_process_memory_bytes(),
                "system": _memory_snapshot(),
                "gpus": self._gpu_snapshot(max_age_s=5.0) if include_gpu else [],
            }

    def list_process_sessions(self) -> Dict[str, Any]:
        from infrastructure.process_session_catalog import list_process_session_groups, is_per_process_audio_supported

        with self._device_lock:
            supported = is_per_process_audio_supported()
            raw_groups = list_process_session_groups() if supported else []
            groups: List[Dict[str, Any]] = []
            sessions: List[Dict[str, Any]] = []
            counter = 0
            self._clear_device_tokens_for_kinds({"process"})

            for group_index, group in enumerate(raw_groups):
                group_label = str(group.get("label") or f"Output {group_index + 1}")
                group_id = str(group.get("id") or f"process-output:{group_index}")
                group_sessions: List[Dict[str, Any]] = []
                for session in group.get("sessions", []):
                    pid = _int_or_zero(session.get("pid"))
                    if pid <= 0:
                        continue
                    label = str(session.get("label") or f"PID {pid}")
                    device_id = f"process:{counter}"
                    counter += 1
                    record = {
                        **session,
                        "id": device_id,
                        "kind": "process",
                        "pid": pid,
                        "label": label,
                        "groupId": group_id,
                        "groupLabel": group_label,
                        "endpointId": str(session.get("endpointId") or group_id),
                        "endpointLabel": str(session.get("endpointLabel") or group_label),
                        "fullLabel": f"{group_label} / {label}",
                    }
                    self._device_tokens[device_id] = (
                        "process",
                        {
                            "pid": pid,
                            "index": session.get("index", 0),
                            "endpointId": record["endpointId"],
                            "endpointLabel": record["endpointLabel"],
                        },
                        str(record["fullLabel"]),
                    )
                    group_sessions.append(record)
                    sessions.append(record)
                if group_sessions:
                    groups.append({"id": group_id, "label": group_label, "sessions": group_sessions})

            return {"sessions": sessions, "groups": groups, "supported": supported}

    def list_models(self, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        from application.model_policy import ASR_MODEL_NAMES

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
        from application.model_download import normalize_model_reference, scan_local_models

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
        from application.model_download import is_model_cached, is_builtin_model
        cached = is_model_cached(name, models_dir=models_dir)
        return {
            "name": name, "label": name,
            "cached": cached, "compatible": cached,
            "status": "compatible" if cached else "recommended",
            "source": "recommended",
            "builtin": is_builtin_model(name),
            "recommended": True, "downloadable": True, "deletable": False,
            **self._download_fields(name, downloads),
        }

    def _custom_model_record(self, name: str, models_dir: Any, downloads: Dict) -> Dict[str, Any]:
        from application.model_download import is_model_cached
        cached = is_model_cached(name, models_dir=models_dir)
        return {
            "name": name, "label": name,
            "cached": cached, "compatible": cached,
            "status": "compatible" if cached else "unknown_remote",
            "source": "custom",
            "builtin": False, "recommended": False, "downloadable": True, "deletable": False,
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
                    "downloadedBytes": _int_or_zero(payload.get("downloadedBytes")),
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
                    "downloadedBytes": _int_or_zero(payload.get("downloadedBytes")),
                    "totalBytes": _int_or_zero(payload.get("totalBytes")),
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
                    "downloadedBytes": _int_or_zero(previous.get("downloadedBytes")),
                    "totalBytes": _int_or_zero(previous.get("totalBytes")),
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
                    "downloadedBytes": _int_or_zero(payload.get("downloadedBytes")),
                    "totalBytes": _int_or_zero(payload.get("totalBytes")),
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
                    "downloadedBytes": _int_or_zero(previous.get("downloadedBytes")),
                    "totalBytes": _int_or_zero(previous.get("totalBytes")),
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

    def save_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        config = params.get("config")
        if not isinstance(config, dict):
            raise ValueError("save_config requires params.config object")
        self.config_repository.write(config)
        try:
            from application.local_paths import configure_project_local_io

            configure_project_local_io(self.project_root, models_dir=self._models_dir(config))
        except Exception:
            pass
        self._catalog_cache_clear()
        return {"saved": True, "config": self._read_config()}

    def list_devices(self) -> Dict[str, Any]:
        with self._device_lock:
            self._clear_device_tokens_for_kinds({"loopback", "input"})
            loopback, loopback_error = self._read_device_group("loopback")
            inputs, input_error = self._read_device_group("input")
            return {
                "loopback": loopback,
                "input": inputs,
                "errors": [error for error in (loopback_error, input_error) if error],
            }

    def add_source(self, params: Dict[str, Any]) -> Dict[str, Any]:
        controller = self._require_session_controller()
        device_id = str(params.get("deviceId", params.get("device_id", "")) or "")
        if not device_id:
            raise ValueError("add_source requires params.deviceId")
        with self._device_lock:
            if device_id not in self._device_tokens:
                if device_id.startswith("process:"):
                    self.list_process_sessions()
                else:
                    self.list_devices()
            token_record = self._device_tokens.get(device_id)
        if token_record is None:
            raise KeyError(f"Unknown device id: {device_id}")
        kind, token, label = token_record
        return controller.add_source(
            kind=kind,
            token=token,
            label=str(params.get("label") or label),
            name=str(params.get("name") or ""),
        )

    def remove_source(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._require_session_controller().remove_source(name=str(params.get("name", "")))

    def set_source_enabled(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._require_session_controller().set_source_enabled(
            name=str(params.get("name", "")),
            enabled=bool(params.get("enabled", False)),
        )

    def set_source_delay(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._require_session_controller().set_source_delay(
            name=str(params.get("name", "")),
            delay_ms=params.get("delayMs", params.get("delay_ms", 0.0)),
        )

    def start_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from application.model_download import is_model_cached
        merged = self._start_params_from_config()
        merged.update(params)
        merged = self._resolve_diarization_start_params(merged)
        controller = self._require_session_controller()
        if bool(merged.get("asrEnabled", False)):
            model_name = str(merged.get("model", "") or "").strip()
            if model_name:
                config = self._read_config()
                models_dir = self._models_dir(config)
                if not is_model_cached(model_name, models_dir=models_dir):
                    return self._download_then_start(controller, model_name, merged, models_dir, config)
        return controller.start_session(merged)

    def _download_then_start(
        self,
        controller: Any,
        model_name: str,
        params: Dict[str, Any],
        models_dir: Any,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        from application.model_download import download_model_async
        models_cfg = config.get("models", {}) if isinstance(config.get("models"), dict) else {}
        proxy = str(models_cfg.get("proxy") or "") if models_cfg.get("use_proxy") else ""

        controller.begin_model_download(model_name)

        def on_progress(info: dict) -> None:
            controller.update_model_download_progress(info)

        def on_done(error: Optional[str]) -> None:
            controller.finish_model_download(error=error or "")
            if not error:
                try:
                    controller.start_session(params)
                except Exception:
                    pass

        download_model_async(model_name, on_progress, on_done, models_dir=models_dir, proxy=proxy)
        return controller.snapshot()

    def stop_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        merged = self._start_params_from_config()
        config = self._read_config()
        ui = config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}
        merged["runOfflinePass"] = bool(ui.get("offline_on_stop", False))
        merged.update(params)
        return self._require_session_controller().stop_session(merged)

    def clear_transcript(self) -> Dict[str, Any]:
        return self._require_session_controller().clear_transcript()

    def invoke_assistant(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.assistant_controller is None:
            raise RuntimeError("Assistant controller is not configured")
        return self.assistant_controller.invoke(params)

    def start_assistant_login(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.assistant_controller is None:
            raise RuntimeError("Assistant controller is not configured")
        return self.assistant_controller.start_provider_login(params)

    def ping_assistant_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.assistant_controller is None:
            raise RuntimeError("Assistant controller is not configured")
        return self.assistant_controller.ping_provider(params)

    def start_local_llm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.assistant_controller is None:
            raise RuntimeError("Assistant controller is not configured")
        return self.assistant_controller.start_local_model(params)

    def stop_local_llm(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.assistant_controller is None:
            raise RuntimeError("Assistant controller is not configured")
        return self.assistant_controller.stop_local_model(params)

    _NO_PARAMS: frozenset = frozenset({
        "get_state", "get_runtime_state", "get_config",
        "list_devices", "clear_transcript", "list_process_sessions",
    })
    _METHODS: frozenset = frozenset({
        "ping", "get_state", "get_runtime_state", "get_config",
        "get_resource_usage", "save_config", "list_devices",
        "add_source", "remove_source", "set_source_enabled", "set_source_delay",
        "start_session", "stop_session", "clear_transcript",
        "invoke_assistant", "start_assistant_login", "ping_assistant_provider",
        "list_models", "download_model", "delete_model", "model_metadata",
        "list_diarization_models", "download_diarization_model", "delete_diarization_model",
        "list_llm_models", "download_llm_model", "delete_llm_model",
        "start_local_llm", "stop_local_llm", "list_process_sessions",
    })

    def handle(self, method: str, params: Dict[str, Any] | None = None) -> Any:
        if method not in self._METHODS:
            raise KeyError(f"Unknown backend method: {method}")
        params = dict(params or {})
        fn = getattr(self, method)
        return fn() if method in self._NO_PARAMS else fn(params)

    def _read_config(self) -> Dict[str, Any]:
        try:
            config = self.config_repository.read()
        except Exception:
            return {}
        return config if isinstance(config, dict) else {}

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
            "downloadedBytes": _int_or_zero(dl.get("downloadedBytes")),
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

    def _asr_model_options(self, config: Dict[str, Any]) -> List[str]:
        local = _scan_compatible_asr_models(self._models_dir(config))
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

    def _start_params_from_config(self) -> Dict[str, Any]:
        config = self._read_config()
        ui = config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}
        asr = config.get("asr", {}) if isinstance(config.get("asr"), dict) else {}
        return {
            "asrEnabled": bool(ui.get("asr_enabled", False)),
            "language": str(ui.get("lang", "")),
            "asrMode": "split" if int(ui.get("asr_mode", 0) or 0) == 1 else "mix",
            "profile": str(ui.get("profile", "")),
            "model": str(ui.get("model", "")),
            "wavEnabled": bool(ui.get("wav_enabled", False)),
            "outputFile": str(ui.get("output_file", "") or ""),
            "realtimeTranscriptToFile": bool(ui.get("rt_transcript_to_file", False)),
            **asr,
        }

    def _resolve_diarization_start_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(params.get("diarizationEnabled", params.get("diarization_enabled", False))):
            return params

        backend = str(params.get("diarBackend", params.get("diar_backend", "online")) or "online").strip().lower()
        sherpa_path = str(
            params.get(
                "diarSherpaEmbeddingModelPath",
                params.get("diar_sherpa_embedding_model_path", ""),
            )
            or ""
        ).strip()

        if backend == "sherpa_onnx" and not sherpa_path:
            return self._with_default_sherpa_model(params)
        if backend == "online" and not _module_available("resemblyzer"):
            return self._with_default_sherpa_model(params)
        return params

    def _with_default_sherpa_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from application.diarization_model_download import default_cached_diarization_model

        model = default_cached_diarization_model(
            project_root=self.project_root,
            models_dir=self._models_dir(),
        )
        path = str(model.get("path") or "") if model else ""
        if not path:
            return params
        next_params = dict(params)
        next_params["diarBackend"] = "sherpa_onnx"
        next_params["diar_backend"] = "sherpa_onnx"
        next_params["diarSherpaEmbeddingModelPath"] = path
        next_params["diar_sherpa_embedding_model_path"] = path
        next_params.setdefault("diarSherpaProvider", str(model.get("provider") or "cpu"))
        next_params.setdefault("diar_sherpa_provider", str(model.get("provider") or "cpu"))
        return next_params

    def _hardware_snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()
        if self._hardware_cache is not None and now - self._hardware_cache_ts < 10.0:
            return dict(self._hardware_cache)

        snapshot = {
            "cpu": {
                "name": _cpu_name(),
                "logicalCores": int(os.cpu_count() or 0),
            },
            "memory": {
                "totalBytes": _physical_memory_bytes(),
            },
            "gpus": self._gpu_snapshot(max_age_s=10.0),
        }
        self._hardware_cache = snapshot
        self._hardware_cache_ts = now
        return dict(snapshot)

    def _gpu_snapshot(self, *, max_age_s: float) -> List[Dict[str, Any]]:
        with self._resource_lock:
            now = time.monotonic()
            if self._gpu_cache_ts > 0.0 and now - self._gpu_cache_ts < max(0.0, float(max_age_s)):
                return [dict(gpu) for gpu in self._gpu_cache]
            gpus = _nvidia_gpu_snapshot()
            self._gpu_cache = [dict(gpu) for gpu in gpus]
            self._gpu_cache_ts = now
            return [dict(gpu) for gpu in self._gpu_cache]

    def _session_snapshot(self) -> Dict[str, Any]:
        if self.session_controller is None:
            return {
                "status": "idle",
                "running": False,
                "state": "idle",
                "asrRunning": False,
                "wavRecording": False,
                "sources": [],
            }
        return self.session_controller.snapshot()

    def _assistant_snapshot(self) -> Dict[str, Any]:
        if self.assistant_controller is None:
            return {
                "enabled": False,
                "providerAvailable": False,
                "providerId": "",
                "providerMessage": "",
                "providerErrorCode": "",
                "providerSuggestion": "",
                "providerAuthRequired": False,
                "providerLoginSupported": False,
                "providerLocalHome": "",
                "busy": False,
                "profiles": [],
                "providers": [],
                "lastResponse": {},
                "lastError": "",
            }
        return self.assistant_controller.snapshot()

    def _require_session_controller(self) -> HeadlessSessionController:
        if self.session_controller is None:
            raise RuntimeError("Session controller is not configured")
        return self.session_controller

    def _read_device_group(self, kind: str) -> tuple[List[Dict[str, Any]], str | None]:
        try:
            if kind == "loopback":
                raw_devices = self.device_catalog.list_loopback_devices()
            else:
                raw_devices = self.device_catalog.list_input_devices()
        except Exception as exc:
            return [], f"{kind}: {type(exc).__name__}: {exc}"

        devices: List[Dict[str, Any]] = []
        for index, item in enumerate(raw_devices):
            label, token = item
            device_id = f"{kind}:{index}"
            self._device_tokens[device_id] = (kind, token, str(label))
            devices.append(
                {
                    "id": device_id,
                    "kind": kind,
                    "label": str(label),
                    "tokenPreview": _safe_token_preview(token),
                }
            )
        return devices, None

    def _clear_device_tokens_for_kinds(self, kinds: set[str]) -> None:
        for device_id, token_record in list(self._device_tokens.items()):
            if token_record[0] in kinds:
                self._device_tokens.pop(device_id, None)


def _per_process_audio_supported() -> bool:
    try:
        from infrastructure.process_session_catalog import is_per_process_audio_supported
        return is_per_process_audio_supported()
    except Exception:
        return False


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _safe_token_preview(token: object) -> str:
    try:
        text = repr(token)
    except Exception:
        return type(token).__name__
    if len(text) > 180:
        return f"{text[:177]}..."
    return text


def _cpu_name() -> str:
    if platform.system().lower() == "windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                text = str(value).strip()
                if text:
                    return text
        except Exception:
            pass
    return platform.processor() or platform.machine() or "CPU"


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _physical_memory_bytes() -> int:
    return _memory_snapshot().get("totalMemoryBytes", 0)


def _memory_snapshot() -> Dict[str, int]:
    if platform.system().lower() == "windows":
        try:
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return {
                    "totalMemoryBytes": int(status.ullTotalPhys),
                    "freeMemoryBytes": int(status.ullAvailPhys),
                }
        except Exception:
            return {"totalMemoryBytes": 0, "freeMemoryBytes": 0}
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return {
            "totalMemoryBytes": int(pages) * int(page_size),
            "freeMemoryBytes": int(available_pages) * int(page_size),
        }
    except Exception:
        return {"totalMemoryBytes": 0, "freeMemoryBytes": 0}


def _current_process_memory_bytes() -> int:
    if platform.system().lower() == "windows":
        try:
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
            psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_ProcessMemoryCounters),
                ctypes.c_ulong,
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except Exception:
            return 0
    try:
        statm = Path("/proc/self/statm")
        if statm.exists():
            pages = int(statm.read_text(encoding="utf-8").split()[1])
            return pages * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        pass
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system().lower() == "darwin" else value * 1024
    except Exception:
        return 0


def _nvidia_gpu_snapshot() -> List[Dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    command = [
        executable,
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True, timeout=2.0)
    except Exception:
        return []
    if completed.returncode != 0:
        return []

    gpus: List[Dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        memory_total_mib = _int_or_zero(parts[1])
        memory_used_mib = _int_or_zero(parts[2])
        gpus.append(
            {
                "name": parts[0],
                "memoryTotalMiB": memory_total_mib,
                "memoryUsedMiB": memory_used_mib,
                "memoryFreeMiB": max(0, memory_total_mib - memory_used_mib),
                "gpuUtilizationPct": _int_or_zero(parts[3]),
                "memoryUtilizationPct": _int_or_zero(parts[4]),
            }
        )
    return gpus


def _scan_compatible_asr_models(models_dir: Any) -> List[str]:
    try:
        from application.model_download import scan_local_models
        return [
            str(m.get("name") or "")
            for m in scan_local_models(models_dir)
            if m.get("compatible") and str(m.get("name") or "").strip()
        ]
    except Exception:
        return []


def _int_or_zero(raw: object) -> int:
    return _int_or_default(raw, 0)


def _int_or_default(raw: object, default: int) -> int:
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)
