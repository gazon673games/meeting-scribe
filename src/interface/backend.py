from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from application.device_catalog import DeviceCatalog
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
            "configSummary": {
                "language": str(ui.get("lang", "")),
                "model": str(ui.get("model", "")),
                "profile": str(ui.get("profile", "")),
                "asrEnabled": bool(ui.get("asr_enabled", False)),
                "asrMode": "split" if int(ui.get("asr_mode", 0) or 0) == 1 else "mix",
                "wavEnabled": bool(ui.get("wav_enabled", False)),
                "offlineOnStop": bool(ui.get("offline_on_stop", False)),
                "computeType": str(asr.get("compute_type", "")),
                "codexEnabled": bool(codex.get("enabled", False)) if isinstance(codex, dict) else False,
                "codexProfiles": len(profiles) if isinstance(profiles, list) else 0,
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
        if method == "set_source_enabled":
            return self.set_source_enabled(params)
        if method == "set_source_delay":
            return self.set_source_delay(params)
        if method == "start_session":
            return self.start_session(params)
        if method == "stop_session":
            return self.stop_session(params)
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
            **asr,
        }

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
