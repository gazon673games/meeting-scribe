from __future__ import annotations

import ctypes
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
from application.asr_profiles import PROFILE_BALANCED, PROFILE_CUSTOM, PROFILE_QUALITY, PROFILE_REALTIME, profile_defaults
from application.device_catalog import DeviceCatalog
from application.model_policy import ASR_MODEL_NAMES
from interface.assistant_controller import AssistantController
from interface.session_controller import HeadlessSessionController
from settings.application.config_repository import ConfigRepository


DeviceToken = Tuple[str, object, str]


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
    _resource_cpu_time_s: float = field(default=0.0, init=False, repr=False)
    _resource_wall_time_s: float = field(default=0.0, init=False, repr=False)
    _downloads: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _download_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
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
                "codexEnabled": bool(codex.get("enabled", False)) if isinstance(codex, dict) else False,
                "codexProfiles": len(profiles) if isinstance(profiles, list) else 0,
            },
            "options": {
                "languages": list(SUPPORTED_ASR_LANGUAGES),
                "asrProfiles": [PROFILE_REALTIME, PROFILE_BALANCED, PROFILE_QUALITY, PROFILE_CUSTOM],
                "asrModels": self._asr_model_options(config),
                "asrModes": [
                    {"id": "mix", "label": "MIX (master)"},
                    {"id": "split", "label": "SPLIT (all sources)"},
                ],
                "asrDevices": ["cuda", "cpu"],
                "computeTypes": ["int8_float16", "float16", "int8", "int8_float32", "float32"],
                "overloadStrategies": ["drop_old", "keep_all"],
                "profileDefaults": {
                    PROFILE_REALTIME: profile_defaults(PROFILE_REALTIME),
                    PROFILE_BALANCED: profile_defaults(PROFILE_BALANCED),
                    PROFILE_QUALITY: profile_defaults(PROFILE_QUALITY),
                    PROFILE_CUSTOM: profile_defaults(PROFILE_BALANCED),
                },
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

    def get_config(self) -> Dict[str, Any]:
        return self._read_config()

    def get_resource_usage(self) -> Dict[str, Any]:
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
            "gpus": _nvidia_gpu_snapshot(),
        }

    def list_process_sessions(self) -> Dict[str, Any]:
        from infrastructure.process_session_catalog import list_process_session_groups, is_per_process_audio_supported

        supported = is_per_process_audio_supported()
        raw_groups = list_process_session_groups() if supported else []
        groups: List[Dict[str, Any]] = []
        sessions: List[Dict[str, Any]] = []
        counter = 0

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
        from application.model_download import (
            is_model_cached,
            is_builtin_model,
            normalize_model_reference,
            scan_local_models,
        )
        from application.model_policy import ASR_MODEL_NAMES
        config = self._read_config()
        models_dir = self._models_dir_from_params(params, config)
        downloads = self._download_snapshot()
        models = []
        recommended_refs = {normalize_model_reference(name) for name in ASR_MODEL_NAMES}
        for name in ASR_MODEL_NAMES:
            cached = is_model_cached(name, models_dir=models_dir)
            models.append({
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
            })
        for record in scan_local_models(models_dir):
            if str(record.get("name") or "") in recommended_refs:
                continue
            models.append({**record, **self._download_fields(str(record.get("name") or ""), downloads)})

        current_model = str((config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}).get("model", "")).strip()
        known_refs = {normalize_model_reference(str(model.get("name") or "")) for model in models}
        if current_model and normalize_model_reference(current_model) not in known_refs:
            cached = is_model_cached(current_model, models_dir=models_dir)
            models.append({
                "name": current_model,
                "label": current_model,
                "cached": cached,
                "compatible": cached,
                "status": "compatible" if cached else "unknown_remote",
                "source": "custom",
                "builtin": False,
                "recommended": False,
                "downloadable": True,
                "deletable": False,
                **self._download_fields(current_model, downloads),
            })

        known_refs = {normalize_model_reference(str(model.get("name") or "")) for model in models}
        for name, dl in downloads.items():
            normalized = normalize_model_reference(name)
            if not normalized or normalized in known_refs:
                continue
            cached = is_model_cached(name, models_dir=models_dir)
            models.append({
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
            })
            known_refs.add(normalized)

        return {"models": models, "modelsDir": str(models_dir), "activeDownloads": self._active_download_count(downloads)}

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
        return {"deleted": True, "name": name}

    def model_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from application.model_download import model_metadata

        name = str(params.get("name") or "").strip()
        if not name:
            raise ValueError("model_metadata requires params.name")
        return model_metadata(name, models_dir=self._models_dir_from_params(params))

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
        return {"saved": True, "config": self._read_config()}

    def list_devices(self) -> Dict[str, Any]:
        self._device_tokens.clear()
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
        merged = self._start_params_from_config()
        merged.update(params)
        return self._require_session_controller().start_session(merged)

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

    def handle(self, method: str, params: Dict[str, Any] | None = None) -> Any:
        params = dict(params or {})
        if method == "ping":
            return self.ping(params)
        if method == "get_state":
            return self.get_state()
        if method == "get_config":
            return self.get_config()
        if method == "get_resource_usage":
            return self.get_resource_usage()
        if method == "save_config":
            return self.save_config(params)
        if method == "list_devices":
            return self.list_devices()
        if method == "add_source":
            return self.add_source(params)
        if method == "remove_source":
            return self.remove_source(params)
        if method == "set_source_enabled":
            return self.set_source_enabled(params)
        if method == "set_source_delay":
            return self.set_source_delay(params)
        if method == "start_session":
            return self.start_session(params)
        if method == "stop_session":
            return self.stop_session(params)
        if method == "clear_transcript":
            return self.clear_transcript()
        if method == "invoke_assistant":
            return self.invoke_assistant(params)
        if method == "list_models":
            return self.list_models(params)
        if method == "download_model":
            return self.download_model(params)
        if method == "delete_model":
            return self.delete_model(params)
        if method == "model_metadata":
            return self.model_metadata(params)
        if method == "list_process_sessions":
            return self.list_process_sessions()
        raise KeyError(f"Unknown backend method: {method}")

    def _read_config(self) -> Dict[str, Any]:
        try:
            config = self.config_repository.read()
        except Exception:
            return {}
        return config if isinstance(config, dict) else {}

    def _download_snapshot(self) -> Dict[str, Dict[str, Any]]:
        with self._download_lock:
            return {str(name): dict(record) for name, record in self._downloads.items()}

    def _download_record(self, name: str) -> Dict[str, Any]:
        from application.model_download import normalize_model_reference

        key = normalize_model_reference(name)
        with self._download_lock:
            return dict(self._downloads.get(name) or self._downloads.get(key) or {})

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

    def _set_download_state(self, name: str, record: Dict[str, Any]) -> None:
        with self._download_lock:
            self._downloads[str(name)] = dict(record)
            active_downloads = self._active_download_count(self._downloads)
        self._emit(
            "model_download_updated",
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

    def _asr_model_options(self, config: Dict[str, Any]) -> List[str]:
        try:
            from application.model_download import scan_local_models

            local = [
                str(model.get("name") or "")
                for model in scan_local_models(self._models_dir(config))
                if model.get("compatible") and str(model.get("name") or "").strip()
            ]
        except Exception:
            local = []
        current = str((config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}).get("model", "") or "").strip()
        out: List[str] = []
        seen: set[str] = set()
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
            "gpus": _nvidia_gpu_snapshot(),
        }
        self._hardware_cache = snapshot
        self._hardware_cache_ts = now
        return dict(snapshot)

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
                "providerMessage": "",
                "providerErrorCode": "",
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


def _per_process_audio_supported() -> bool:
    try:
        from infrastructure.process_session_catalog import is_per_process_audio_supported
        return is_per_process_audio_supported()
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


def _int_or_zero(raw: object) -> int:
    return _int_or_default(raw, 0)


def _int_or_default(raw: object, default: int) -> int:
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)
