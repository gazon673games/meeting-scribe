from __future__ import annotations

import os
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from application import model_policy
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
from application.native_asr_models import (
    GIGAAM_MODEL_ID,
    NEMOTRON_MODEL_ID,
    PARAKEET_MODEL_ID,
    QWEN_ASR_MODEL_ID,
    is_native_asr_model_cached,
)
from interface.backend_parts.system_utils import (
    cpu_name,
    current_process_memory_bytes,
    int_or_default,
    memory_snapshot,
    per_process_audio_supported,
    physical_memory_bytes,
)
from shared.domain.mappings import first_param

if TYPE_CHECKING:
    from interface.backend_impl import ElectronBackend


def _ui_model(config: dict[str, Any]) -> str:
    ui = config.get('ui', {}) if isinstance(config.get('ui'), dict) else {}
    return str(ui.get('model', '') or '').strip()


def set_event_sink(backend: ElectronBackend, event_sink) -> None:  # noqa: ANN001
    backend._event_sink = event_sink
    if backend.session_controller is not None:
        backend.session_controller.set_event_sink(event_sink)
    if backend.assistant_controller is not None:
        backend.assistant_controller.set_event_sink(event_sink)


def ping(
    backend: ElectronBackend, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {'pong': True, 'echo': dict(params or {})}


def get_state(backend: ElectronBackend) -> dict[str, Any]:
    config = backend._read_config()
    ui = config.get('ui', {}) if isinstance(config.get('ui'), dict) else {}
    asr = config.get('asr', {}) if isinstance(config.get('asr'), dict) else {}
    codex = (
        config.get('codex', {}) if isinstance(config.get('codex'), dict) else {}
    )
    profiles = codex.get('profiles', []) if isinstance(codex, dict) else []
    session = backend._session_snapshot()
    assistant = backend._assistant_snapshot()
    asr_models = backend._asr_model_options(config)
    policy_models = [
        model
        for model in asr_models
        if model != NEMOTRON_MODEL_ID
        or is_native_asr_model_cached(
            model, models_dir=backend._models_dir_from_params(config=config)
        )
    ]
    current_model = str(ui.get('model', '') or '').strip()
    language = str(ui.get('lang', '') or '')

    return {
        'protocolVersion': backend.protocol_version,
        'appName': 'Meeting Scribe',
        'session': session,
        'assistant': assistant,
        'paths': {
            'projectRoot': str(backend.project_root),
            'config': str(Path(backend.config_repository.path)),  # type: ignore[attr-defined]
            'models': str(backend._models_dir()),
        },
        'hardware': backend._hardware_snapshot(),
        'configSummary': _config_summary(
            ui=ui, asr=asr, codex=codex, profiles=profiles
        ),
        'options': _state_options(
            asr_models=asr_models,
            policy_models=policy_models,
            current_model=current_model,
            language=language,
        ),
        'capabilities': {
            'config': True,
            'devices': True,
            'sourceControl': backend.session_controller is not None,
            'sessionControl': backend.session_controller is not None,
            'assistant': backend.assistant_controller is not None,
            'liveTranscript': backend.session_controller is not None,
            'perProcessAudio': per_process_audio_supported(),
        },
    }


def get_runtime_state(backend: ElectronBackend) -> dict[str, Any]:
    return {
        'protocolVersion': backend.protocol_version,
        'session': backend._session_snapshot(),
    }


def get_config(backend: ElectronBackend) -> dict[str, Any]:
    return backend._read_config()


def get_resource_usage(
    backend: ElectronBackend, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    params = params if isinstance(params, dict) else {}
    include_gpu = bool(
        first_param(params, 'includeGpu', 'include_gpu', default=True)
    )
    with backend._resource_lock:
        now = time.monotonic()
        cpu_time_s = time.process_time()
        cpu_pct = 0.0
        if backend._resource_wall_time_s > 0.0:
            wall_delta = max(1e-6, now - backend._resource_wall_time_s)
            cpu_delta = max(0.0, cpu_time_s - backend._resource_cpu_time_s)
            cpu_pct = min(
                100.0,
                (cpu_delta / wall_delta / max(1, os.cpu_count() or 1)) * 100.0,
            )
        backend._resource_wall_time_s = now
        backend._resource_cpu_time_s = cpu_time_s
        return {
            'pid': os.getpid(),
            'cpuPct': cpu_pct,
            'memoryBytes': current_process_memory_bytes(),
            'system': memory_snapshot(),
            'gpus': backend._gpu_snapshot(max_age_s=5.0) if include_gpu else [],
        }


def _hardware_snapshot(backend: ElectronBackend) -> dict[str, Any]:
    now = time.monotonic()
    if (
        backend._hardware_cache is not None
        and now - backend._hardware_cache_ts < 10.0
    ):
        return deepcopy(backend._hardware_cache)

    snapshot = {
        'cpu': {
            'name': cpu_name(),
            'logicalCores': int(os.cpu_count() or 0),
        },
        'memory': {
            'totalBytes': physical_memory_bytes(),
        },
        'gpus': backend._gpu_snapshot(max_age_s=10.0),
    }
    backend._hardware_cache = snapshot
    backend._hardware_cache_ts = now
    return deepcopy(snapshot)


def _session_snapshot(backend: ElectronBackend) -> dict[str, Any]:
    if backend.session_controller is None:
        return {
            'status': 'idle',
            'running': False,
            'state': 'idle',
            'asrRunning': False,
            'wavRecording': False,
            'sources': [],
        }
    return backend.session_controller.snapshot()


def _assistant_snapshot(backend: ElectronBackend) -> dict[str, Any]:
    if backend.assistant_controller is None:
        return {
            'enabled': False,
            'providerAvailable': False,
            'providerId': '',
            'providerMessage': '',
            'providerErrorCode': '',
            'providerSuggestion': '',
            'providerAuthRequired': False,
            'providerLoginSupported': False,
            'providerLocalHome': '',
            'busy': False,
            'profiles': [],
            'providers': [],
            'lastResponse': {},
            'lastError': '',
        }
    return backend.assistant_controller.snapshot()


class BackendStateMixin:
    """Compatibility facade for callers using the former mixin API."""

    set_event_sink = set_event_sink
    ping = ping
    get_state = get_state
    get_runtime_state = get_runtime_state
    get_config = get_config
    get_resource_usage = get_resource_usage
    _hardware_snapshot = _hardware_snapshot
    _session_snapshot = _session_snapshot
    _assistant_snapshot = _assistant_snapshot


def _config_summary(
    *, ui: Any, asr: Any, codex: Any, profiles: Any
) -> dict[str, Any]:
    return {
        'language': str(ui.get('lang', '')),
        'model': str(ui.get('model', '')),
        'profile': str(ui.get('profile', '')),
        'asrEnabled': bool(ui.get('asr_enabled', False)),
        'asrMode': 'split' if int(ui.get('asr_mode', 0) or 0) == 1 else 'mix',
        'wavEnabled': bool(ui.get('wav_enabled', False)),
        'offlineOnStop': bool(ui.get('offline_on_stop', False)),
        'device': str(asr.get('device', '')),
        'computeType': str(asr.get('compute_type', '')),
        'cpuThreads': int_or_default(asr.get('cpu_threads', 0), 0),
        'numWorkers': int_or_default(asr.get('num_workers', 1), 1),
        'diarizationEnabled': bool(asr.get('diarization_enabled', False)),
        'diarBackend': str(asr.get('diar_backend', 'online') or 'online'),
        'codexEnabled': bool(codex.get('enabled', False))
        if isinstance(codex, dict)
        else False,
        'codexProfiles': len(profiles) if isinstance(profiles, list) else 0,
    }


def _state_options(
    *,
    asr_models: Any,
    policy_models: Any,
    current_model: Any,
    language: Any,
) -> dict[str, Any]:
    return {
        'languages': list(SUPPORTED_ASR_LANGUAGES),
        'asrProfiles': [
            PROFILE_ULTRA_FAST,
            PROFILE_REALTIME,
            PROFILE_BALANCED,
            PROFILE_QUALITY,
            PROFILE_CUSTOM,
        ],
        'asrModels': asr_models,
        'offlineAsrModels': [
            PARAKEET_MODEL_ID,
            QWEN_ASR_MODEL_ID,
            GIGAAM_MODEL_ID,
            'bzikst/faster-whisper-large-v3-russian',
            'large-v3',
        ],
        'profileRecommendedModels': {
            profile: model_policy.recommend_asr_model(
                asr_profile=profile,
                language=language,
                current_model=current_model,
                available_models=policy_models,
            )
            for profile in [
                PROFILE_ULTRA_FAST,
                PROFILE_REALTIME,
                PROFILE_BALANCED,
                PROFILE_QUALITY,
            ]
        },
        'asrModes': [
            {'id': 'mix', 'label': 'MIX (master)'},
            {'id': 'split', 'label': 'SPLIT (all sources)'},
        ],
        'asrDevices': ['cuda', 'cpu'],
        'computeTypes': [
            'int8_float16',
            'float16',
            'int8',
            'int8_float32',
            'float32',
        ],
        'diarizationBackends': ['online', 'sherpa_onnx', 'nemo', 'pyannote'],
        'diarizationProviders': ['cpu', 'cuda'],
        'overloadStrategies': ['drop_old', 'keep_all'],
        'profileDefaults': {
            PROFILE_ULTRA_FAST: profile_defaults(PROFILE_ULTRA_FAST),
            PROFILE_REALTIME: profile_defaults(PROFILE_REALTIME),
            PROFILE_BALANCED: profile_defaults(PROFILE_BALANCED),
            PROFILE_QUALITY: profile_defaults(PROFILE_QUALITY),
            PROFILE_CUSTOM: profile_defaults(PROFILE_BALANCED),
        },
        'streamingLockedProfiles': [
            profile
            for profile in [
                PROFILE_ULTRA_FAST,
                PROFILE_REALTIME,
                PROFILE_QUALITY,
            ]
            if profile_requires_streaming(profile)
        ],
    }
