from __future__ import annotations

from typing import Any, Dict, List

from interface.backend_parts.session_dispatch import NO_PARAMS_METHODS, SESSION_METHODS
from interface.backend_parts.session_orchestration import (
    download_then_start,
    resolve_diarization_start_params,
    start_params_from_config,
)
from interface.backend_parts.process_sessions import build_process_session_payload
from interface.backend_parts.system_utils import safe_token_preview


class BackendSessionMixin:
    def list_process_sessions(self) -> Dict[str, Any]:
        from infrastructure.process_session_catalog import is_per_process_audio_supported, list_process_session_groups

        with self._device_lock:
            supported = is_per_process_audio_supported()
            if not supported:
                self._clear_device_tokens_for_kinds({"process"})
                return {"sessions": [], "groups": [], "supported": False}
            payload = build_process_session_payload(self, list_process_session_groups())
            return {**payload, "supported": True}

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
        return self.get_state()

    def list_devices(self) -> Dict[str, Any]:
        with self._device_lock:
            self._clear_device_tokens_for_kinds({"loopback", "input"})
            loopback, loopback_error = self._read_device_group("loopback")
            inputs, input_error = self._read_device_group("input")
            return {
                "loopback": loopback,
                "input": inputs,
                "errors": [error for error in [loopback_error, input_error] if error],
            }

    def add_source(self, params: Dict[str, Any]) -> Dict[str, Any]:
        controller = self._require_session_controller()
        device_id = str(params.get("deviceId", params.get("device_id", "")) or "").strip()
        if not device_id:
            raise ValueError("add_source requires params.deviceId")
        with self._device_lock:
            token_record = self._device_tokens.get(device_id)
        if token_record is None:
            raise KeyError(f"Unknown deviceId: {device_id}")
        kind, token, default_label = token_record
        label = str(params.get("label", default_label) or default_label)
        name = str(params.get("name", "") or "")
        return controller.add_source(kind=kind, token=token, label=label, name=name)

    def remove_source(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._require_session_controller().remove_source(name=str(params.get("name", "")))

    def set_source_enabled(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._require_session_controller().set_source_enabled(
            name=str(params.get("name", "")),
            enabled=bool(params.get("enabled", True)),
        )

    def set_source_delay(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self._require_session_controller().set_source_delay(
            name=str(params.get("name", "")),
            delay_ms=params.get("delayMs", params.get("delay_ms", 0)),
        )

    def start_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        controller = self._require_session_controller()
        config_params = self._start_params_from_config()
        merged = {**config_params, **dict(params)}
        merged = self._resolve_diarization_start_params(merged)
        if bool(merged.get("asrEnabled", merged.get("asr_enabled", False))):
            if bool(params.get("skipModelDownload", params.get("skip_model_download", False))):
                return controller.start_session(merged)
            return self._download_then_start(controller, merged)
        return controller.start_session(merged)

    def _download_then_start(self, controller, merged: Dict[str, Any]) -> Dict[str, Any]:  # noqa: ANN001
        return download_then_start(self, controller, merged)

    def stop_session(self, params: Dict[str, Any]) -> Dict[str, Any]:
        controller = self._require_session_controller()
        config = self._read_config()
        ui = config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}
        merged = {
            "runOfflinePass": bool(ui.get("offline_on_stop", False)),
            **dict(params),
        }
        return controller.stop_session(merged)

    def clear_transcript(self) -> Dict[str, Any]:
        return self._require_session_controller().clear_transcript()

    def invoke_assistant(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.assistant_controller is None:
            raise RuntimeError("Assistant controller is not configured")
        return self.assistant_controller.invoke(params)

    def start_assistant_login(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.assistant_controller is None:
            raise RuntimeError("Assistant controller is not configured")
        return self.assistant_controller.start_login(params)

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

    _NO_PARAMS: frozenset = NO_PARAMS_METHODS
    _METHODS: frozenset = SESSION_METHODS

    def handle(self, method: str, params: Dict[str, Any] | None = None) -> Any:
        if method not in self._METHODS:
            raise KeyError(f"Unknown backend method: {method}")
        params = dict(params or {})
        fn = getattr(self, method)
        return fn() if method in self._NO_PARAMS else fn(params)

    def _start_params_from_config(self) -> Dict[str, Any]:
        return start_params_from_config(self)

    def _resolve_diarization_start_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_diarization_start_params(self, params)

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
                    "tokenPreview": safe_token_preview(token),
                }
            )
        return devices, None

    def _clear_device_tokens_for_kinds(self, kinds: set[str]) -> None:
        for device_id, token_record in list(self._device_tokens.items()):
            if token_record[0] in kinds:
                self._device_tokens.pop(device_id, None)
