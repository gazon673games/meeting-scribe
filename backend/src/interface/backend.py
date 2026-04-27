from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

    protocol_version: int = 1

    def set_event_sink(self, event_sink) -> None:  # noqa: ANN001
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
                "asrModels": list(ASR_MODEL_NAMES),
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
            },
        }

    def get_config(self) -> Dict[str, Any]:
        return self._read_config()

    def save_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        config = params.get("config")
        if not isinstance(config, dict):
            raise ValueError("save_config requires params.config object")
        self.config_repository.write(config)
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
        raise KeyError(f"Unknown backend method: {method}")

    def _read_config(self) -> Dict[str, Any]:
        try:
            config = self.config_repository.read()
        except Exception:
            return {}
        return config if isinstance(config, dict) else {}

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
            return {"enabled": False, "busy": False, "profiles": [], "lastResponse": {}, "lastError": ""}
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


def _physical_memory_bytes() -> int:
    if platform.system().lower() == "windows":
        try:
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return int(status.ullTotalPhys)
        except Exception:
            return 0
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
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
        gpus.append(
            {
                "name": parts[0],
                "memoryTotalMiB": _int_or_zero(parts[1]),
                "memoryUsedMiB": _int_or_zero(parts[2]),
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
