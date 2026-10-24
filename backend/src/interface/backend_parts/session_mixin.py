from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from application.model_directories import model_directory
from interface.backend_parts.process_sessions import (
    build_process_session_payload,
)
from interface.backend_parts.session_dispatch import (
    NO_PARAMS_METHODS,
    SESSION_METHODS,
    dispatch_request,
)
from interface.backend_parts.session_orchestration import (
    download_then_start,
    resolve_diarization_start_params,
    start_params_from_config,
)
from interface.backend_parts.system_utils import safe_token_preview
from shared.domain.mappings import first_param

_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from interface.backend_impl import ElectronBackend


def list_process_sessions(backend: ElectronBackend) -> dict[str, Any]:
    from infrastructure.process_session_catalog import (
        is_per_process_audio_supported,
        list_process_session_groups,
    )

    with backend._device_lock:
        supported = is_per_process_audio_supported()
        if not supported:
            backend._clear_device_tokens_for_kinds({'process'})
            return {'sessions': [], 'groups': [], 'supported': False}
        payload = build_process_session_payload(
            backend, list_process_session_groups()
        )
        return {**payload, 'supported': True}


def save_config(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    config = params.get('config')
    if not isinstance(config, dict):
        raise ValueError('save_config requires params.config object')
    for kind in ('asr', 'diarization', 'llm'):
        model_directory(backend.project_root, config, kind)
    backend.config_repository.write(config)
    try:
        from application.local_paths import configure_project_local_io

        configure_project_local_io(
            backend.project_root,
            models_dir=backend._models_dir_from_params(config=config),
        )
    except Exception:
        _logger.debug(
            'Best-effort operation failed in save_config', exc_info=True
        )
    backend._catalog_cache_clear()
    return backend.get_state()


def list_devices(backend: ElectronBackend) -> dict[str, Any]:
    with backend._device_lock:
        backend._clear_device_tokens_for_kinds({'loopback', 'input'})
        loopback, loopback_error = backend._read_device_group('loopback')
        inputs, input_error = backend._read_device_group('input')
        return {
            'loopback': loopback,
            'input': inputs,
            'errors': [
                error for error in [loopback_error, input_error] if error
            ],
        }


def add_source(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    controller = backend._require_session_controller()
    device_id = str(
        first_param(params, 'deviceId', 'device_id', default='') or ''
    ).strip()
    if not device_id:
        raise ValueError('add_source requires params.deviceId')
    with backend._device_lock:
        token_record = backend._device_tokens.get(device_id)
    if token_record is None:
        raise KeyError(f'Unknown deviceId: {device_id}')
    kind, token, default_label = token_record
    label = str(params.get('label', default_label) or default_label)
    name = str(params.get('name', '') or '')
    return controller.add_source(
        kind=kind, token=token, label=label, name=name, device_id=device_id
    )


def remove_source(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    return backend._require_session_controller().remove_source(
        name=str(params.get('name', ''))
    )


def set_source_enabled(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    return backend._require_session_controller().set_source_enabled(
        name=str(params.get('name', '')),
        enabled=bool(params.get('enabled', True)),
    )


def set_source_delay(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    return backend._require_session_controller().set_source_delay(
        name=str(params.get('name', '')),
        delay_ms=first_param(params, 'delayMs', 'delay_ms', default=0),
    )


def start_session(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    controller = backend._require_session_controller()
    config_params = backend._start_params_from_config()
    merged = {**config_params, **dict(params)}
    merged['asrModelsDir'] = str(backend._models_dir_from_params(merged))
    merged = backend._resolve_diarization_start_params(merged)
    if bool(first_param(merged, 'asrEnabled', 'asr_enabled', default=False)):
        if bool(
            first_param(
                params,
                'skipModelDownload',
                'skip_model_download',
                default=False,
            )
        ):
            return controller.start_session(merged)
        return backend._download_then_start(controller, merged)
    return controller.start_session(merged)


def _download_then_start(
    backend: ElectronBackend, controller, merged: dict[str, Any]
) -> dict[str, Any]:  # noqa: ANN001
    return download_then_start(backend, controller, merged)


def stop_session(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    controller = backend._require_session_controller()
    config = backend._read_config()
    ui = config.get('ui', {}) if isinstance(config.get('ui'), dict) else {}
    asr = config.get('asr', {}) if isinstance(config.get('asr'), dict) else {}
    merged = {
        'runOfflinePass': bool(ui.get('offline_on_stop', False)),
        'offlineModelName': str(
            asr.get('offline_model', ui.get('model', 'large-v3')) or 'large-v3'
        ),
        **dict(params),
    }
    merged['asrModelsDir'] = str(backend._models_dir_from_params(merged))
    return controller.stop_session(merged)


def clear_transcript(backend: ElectronBackend) -> dict[str, Any]:
    return backend._require_session_controller().clear_transcript()


def invoke_assistant(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    if backend.assistant_controller is None:
        raise RuntimeError('Assistant controller is not configured')
    return backend.assistant_controller.invoke(params)


def start_assistant_login(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    if backend.assistant_controller is None:
        raise RuntimeError('Assistant controller is not configured')
    return backend.assistant_controller.start_login(params)


def ping_assistant_provider(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    if backend.assistant_controller is None:
        raise RuntimeError('Assistant controller is not configured')
    return backend.assistant_controller.ping_provider(params)


def start_local_llm(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    if backend.assistant_controller is None:
        raise RuntimeError('Assistant controller is not configured')
    return backend.assistant_controller.start_local_model(params)


def stop_local_llm(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    if backend.assistant_controller is None:
        raise RuntimeError('Assistant controller is not configured')
    return backend.assistant_controller.stop_local_model(params)


def handle(
    backend: ElectronBackend, method: str, params: dict[str, Any] | None = None
) -> Any:
    if backend._shutdown.is_set():
        raise RuntimeError('Backend is shutting down')
    return dispatch_request(backend, method, params)


def _start_params_from_config(backend: ElectronBackend) -> dict[str, Any]:
    return start_params_from_config(backend)


def _resolve_diarization_start_params(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    return resolve_diarization_start_params(backend, params)


def _read_device_group(
    backend: ElectronBackend, kind: str
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        if kind == 'loopback':
            raw_devices = backend.device_catalog.list_loopback_devices()
        else:
            raw_devices = backend.device_catalog.list_input_devices()
    except Exception as error:
        return [], f'{kind}: {type(error).__name__}: {error}'

    devices: list[dict[str, Any]] = []
    for index, item in enumerate(raw_devices):
        label, token = item
        device_id = f'{kind}:{index}'
        backend._device_tokens[device_id] = (kind, token, str(label))
        devices.append(
            {
                'id': device_id,
                'kind': kind,
                'label': str(label),
                'tokenPreview': safe_token_preview(token),
            }
        )
    return devices, None


def _clear_device_tokens_for_kinds(
    backend: ElectronBackend, kinds: set[str]
) -> None:
    for device_id, token_record in list(backend._device_tokens.items()):
        if token_record[0] in kinds:
            backend._device_tokens.pop(device_id, None)


class BackendSessionMixin:
    """Compatibility facade for callers using the former mixin API."""

    _NO_PARAMS: frozenset = NO_PARAMS_METHODS
    _METHODS: frozenset = SESSION_METHODS
    list_process_sessions = list_process_sessions
    save_config = save_config
    list_devices = list_devices
    add_source = add_source
    remove_source = remove_source
    set_source_enabled = set_source_enabled
    set_source_delay = set_source_delay
    start_session = start_session
    _download_then_start = _download_then_start
    stop_session = stop_session
    clear_transcript = clear_transcript
    invoke_assistant = invoke_assistant
    start_assistant_login = start_assistant_login
    ping_assistant_provider = ping_assistant_provider
    start_local_llm = start_local_llm
    stop_local_llm = stop_local_llm
    handle = handle
    _start_params_from_config = _start_params_from_config
    _resolve_diarization_start_params = _resolve_diarization_start_params
    _read_device_group = _read_device_group
    _clear_device_tokens_for_kinds = _clear_device_tokens_for_kinds
