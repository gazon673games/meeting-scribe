from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

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
from interface.backend_parts.system_utils import (
    cpu_name,
    current_process_memory_bytes,
    int_or_default,
    memory_snapshot,
    per_process_audio_supported,
    physical_memory_bytes,
)


def _ui_model(config: Dict[str, Any]) -> str:
    ui = config.get("ui", {}) if isinstance(config.get("ui"), dict) else {}
    return str(ui.get("model", "") or "").strip()


class BackendStateMixin:
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
                "cpuThreads": int_or_default(asr.get("cpu_threads", 0), 0),
                "numWorkers": int_or_default(asr.get("num_workers", 1), 1),
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
                "perProcessAudio": per_process_audio_supported(),
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
                "memoryBytes": current_process_memory_bytes(),
                "system": memory_snapshot(),
                "gpus": self._gpu_snapshot(max_age_s=5.0) if include_gpu else [],
            }

    def _hardware_snapshot(self) -> Dict[str, Any]:
        now = time.monotonic()
        if self._hardware_cache is not None and now - self._hardware_cache_ts < 10.0:
            return dict(self._hardware_cache)

        snapshot = {
            "cpu": {
                "name": cpu_name(),
                "logicalCores": int(os.cpu_count() or 0),
            },
            "memory": {
                "totalBytes": physical_memory_bytes(),
            },
            "gpus": self._gpu_snapshot(max_age_s=10.0),
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

