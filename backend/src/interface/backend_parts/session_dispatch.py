from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class BackendMethod:
    name: str
    accepts_params: bool = True
    serial: bool = False


_METHODS = (
    BackendMethod('ping'),
    BackendMethod('get_state', accepts_params=False),
    BackendMethod('get_runtime_state', accepts_params=False),
    BackendMethod('get_config', accepts_params=False),
    BackendMethod('get_resource_usage'),
    BackendMethod('save_config'),
    BackendMethod('list_devices', accepts_params=False),
    BackendMethod('add_source'),
    BackendMethod('remove_source'),
    BackendMethod('set_source_enabled'),
    BackendMethod('set_source_delay'),
    BackendMethod('start_session', serial=True),
    BackendMethod('stop_session', serial=True),
    BackendMethod('clear_transcript', accepts_params=False),
    BackendMethod('invoke_assistant'),
    BackendMethod('start_assistant_login'),
    BackendMethod('ping_assistant_provider'),
    BackendMethod('list_models'),
    BackendMethod('download_model'),
    BackendMethod('delete_model'),
    BackendMethod('model_metadata'),
    BackendMethod('list_diarization_models'),
    BackendMethod('download_diarization_model'),
    BackendMethod('delete_diarization_model'),
    BackendMethod('list_llm_models'),
    BackendMethod('download_llm_model'),
    BackendMethod('delete_llm_model'),
    BackendMethod('start_local_llm'),
    BackendMethod('stop_local_llm'),
    BackendMethod('list_process_sessions', accepts_params=False),
)

BACKEND_METHODS = MappingProxyType({method.name: method for method in _METHODS})
SESSION_METHODS = frozenset(BACKEND_METHODS)
NO_PARAMS_METHODS = frozenset(
    method.name for method in _METHODS if not method.accepts_params
)
SERIAL_LIFECYCLE_METHODS = frozenset(
    method.name for method in _METHODS if method.serial
)


def dispatch_request(
    backend: Any, method: str, params: dict[str, Any] | None
) -> Any:
    definition = BACKEND_METHODS.get(method)
    if definition is None:
        raise KeyError(f'Unknown backend method: {method}')
    request_params = dict(params or {})
    handler = getattr(backend, definition.name)
    return handler(request_params) if definition.accepts_params else handler()
