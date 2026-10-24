from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from application.model_directories import ModelKind, model_directory
from interface.session_controller import HeadlessSessionController

_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from interface.backend_impl import ElectronBackend


def _read_config(backend: ElectronBackend) -> dict[str, Any]:
    try:
        config = backend.config_repository.read()
    except (OSError, ValueError):
        _logger.warning('Unable to read backend configuration', exc_info=True)
        return {}
    return config if isinstance(config, dict) else {}


def _emit(
    backend: ElectronBackend, event_type: str, payload: dict[str, Any]
) -> None:
    sink = backend._event_sink
    if sink is None:
        return
    try:
        sink(event_type, {'ts': time.time(), **payload})
    except Exception:
        _logger.debug('Best-effort operation failed in _emit', exc_info=True)


def _models_dir(
    backend: ElectronBackend, config: dict[str, Any] | None = None
) -> Path:
    cfg = config if isinstance(config, dict) else backend._read_config()
    models = (
        cfg.get('models', {}) if isinstance(cfg.get('models'), dict) else {}
    )
    raw = str(models.get('cache_dir', '') or '').strip()
    return (
        Path(raw).expanduser().resolve()
        if raw
        else Path(backend.project_root).resolve() / 'models'
    )


def _models_dir_from_params(
    backend: ElectronBackend,
    params: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    *,
    kind: ModelKind = 'asr',
) -> Path:
    return model_directory(
        backend.project_root,
        config if config is not None else backend._read_config(),
        kind,
        params,
    )


def _llm_models_dir_from_params(
    backend: ElectronBackend, params: dict[str, Any] | None = None
) -> Path:
    return model_directory(
        backend.project_root, backend._read_config(), 'llm', params
    )


def _require_session_controller(
    backend: ElectronBackend,
) -> HeadlessSessionController:
    if backend.session_controller is None:
        raise RuntimeError('Session controller is not configured')
    return backend.session_controller


class BackendBaseMixin:
    """Compatibility facade for callers using the former mixin API."""

    _read_config = _read_config
    _emit = _emit
    _models_dir = _models_dir
    _models_dir_from_params = _models_dir_from_params
    _llm_models_dir_from_params = _llm_models_dir_from_params
    _require_session_controller = _require_session_controller
