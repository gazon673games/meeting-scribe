from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from application.asr_profiles import PROFILE_ULTRA_FAST
from interface.backend_parts.system_utils import int_or_zero, module_available, per_process_audio_supported, safe_token_preview


class BackendSessionMixin:
    def list_process_sessions(self) -> Dict[str, Any]:
        from infrastructure.process_session_catalog import is_per_process_audio_supported, list_process_session_groups

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
                    pid = int_or_zero(session.get("pid"))
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
        from application.model_download import is_model_cached, normalize_model_reference

        model_name = str(merged.get("model", "")).strip()
        if not model_name:
            return controller.start_session(merged)
        normalized_model = normalize_model_reference(model_name)
        if not is_model_cached(normalized_model, models_dir=self._models_dir()):
            controller.begin_model_download(normalized_model)
            try:
                self.download_model({"name": normalized_model, "modelsDir": str(self._models_dir())})
                wait_limit_s = float(merged.get("downloadWaitTimeoutS", merged.get("download_wait_timeout_s", 3600)))
                start = time.monotonic()
                while time.monotonic() - start < wait_limit_s:
                    state = self._download_record(normalized_model)
                    if state.get("state") == "done":
                        break
                    if state.get("state") == "error":
                        raise RuntimeError(str(state.get("error") or "ASR model download failed"))
                    time.sleep(0.2)
                else:
                    raise RuntimeError(f"Timed out while downloading model '{normalized_model}'")
            except Exception as exc:
                controller.finish_model_download(str(exc))
                raise
            controller.finish_model_download("")

        return controller.start_session(merged)

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

    _NO_PARAMS: frozenset = frozenset({
        "get_state",
        "get_runtime_state",
        "get_config",
        "list_devices",
        "clear_transcript",
        "list_process_sessions",
    })
    _METHODS: frozenset = frozenset({
        "ping",
        "get_state",
        "get_runtime_state",
        "get_config",
        "get_resource_usage",
        "save_config",
        "list_devices",
        "add_source",
        "remove_source",
        "set_source_enabled",
        "set_source_delay",
        "start_session",
        "stop_session",
        "clear_transcript",
        "invoke_assistant",
        "start_assistant_login",
        "ping_assistant_provider",
        "list_models",
        "download_model",
        "delete_model",
        "model_metadata",
        "list_diarization_models",
        "download_diarization_model",
        "delete_diarization_model",
        "list_llm_models",
        "download_llm_model",
        "delete_llm_model",
        "start_local_llm",
        "stop_local_llm",
        "list_process_sessions",
    })

    def handle(self, method: str, params: Dict[str, Any] | None = None) -> Any:
        if method not in self._METHODS:
            raise KeyError(f"Unknown backend method: {method}")
        params = dict(params or {})
        fn = getattr(self, method)
        return fn() if method in self._NO_PARAMS else fn(params)

    def _start_params_from_config(self) -> Dict[str, Any]:
        config = self._read_config()
        ui = config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}
        asr = config.get("asr", {}) if isinstance(config.get("asr"), dict) else {}
        speaker_identity = (
            config.get("speaker_identity", {})
            if isinstance(config.get("speaker_identity"), dict)
            else {}
        )
        return {
            "asrEnabled": bool(ui.get("asr_enabled", False)),
            "language": str(ui.get("lang", "")),
            "asrMode": "split" if int(ui.get("asr_mode", 0) or 0) == 1 else "mix",
            "profile": str(ui.get("profile", "")),
            "model": str(ui.get("model", "")),
            "wavEnabled": bool(ui.get("wav_enabled", False)),
            "outputFile": str(ui.get("output_file", "") or ""),
            "realtimeTranscriptToFile": bool(ui.get("rt_transcript_to_file", False)),
            "speakerIdentity": dict(speaker_identity),
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
        if backend == "online" and not module_available("resemblyzer"):
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
