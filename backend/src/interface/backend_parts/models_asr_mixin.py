from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from application.model_policy import ASR_MODEL_NAMES
from interface.backend_parts.system_utils import (
    int_or_zero,
    scan_compatible_asr_models,
)
from shared.domain.mappings import first_param

if TYPE_CHECKING:
    from interface.backend_impl import ElectronBackend


def _ui_model(config: dict[str, Any]) -> str:
    ui = config.get('ui', {}) if isinstance(config.get('ui'), dict) else {}
    return str(ui.get('model', '') or '').strip()


def list_models(
    backend: ElectronBackend, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    config = backend._read_config()
    models_dir = backend._models_dir_from_params(params, config)
    downloads = backend._download_snapshot()
    current_model = _ui_model(config)
    cache_key = ('asr', str(models_dir), current_model)

    if not backend._active_download_count(downloads):
        hit = backend._catalog_cache_get(cache_key)
        if hit is not None:
            return hit

    models = backend._build_model_list(
        ASR_MODEL_NAMES, models_dir, downloads, current_model
    )
    result = {
        'models': models,
        'modelsDir': str(models_dir),
        'activeDownloads': backend._active_download_count(downloads),
    }
    if not result['activeDownloads']:
        backend._catalog_cache_put(cache_key, result)
    return result


def _build_model_list(
    backend: ElectronBackend,
    recommended_names: tuple[str, ...] | list[str],
    models_dir: Path,
    downloads: dict[str, dict[str, Any]],
    current_model: str,
) -> list[dict[str, Any]]:
    from application.model_download import normalize_model_reference

    recommended_refs = {normalize_model_reference(n) for n in recommended_names}
    models: list[dict[str, Any]] = [
        backend._recommended_model_record(n, models_dir, downloads)
        for n in recommended_names
    ]
    models += backend._local_model_records(
        models_dir, downloads, recommended_refs
    )
    backend._add_unlisted_models(models, downloads, current_model, models_dir)
    return models


def _local_model_records(
    backend: ElectronBackend,
    models_dir: Path,
    downloads: dict[str, dict[str, Any]],
    skip_refs: set[str],
) -> list[dict[str, Any]]:
    from application.model_download import (
        normalize_model_reference,
        scan_local_models,
    )

    result = []
    for rec in scan_local_models(models_dir):
        name = str(rec.get('name') or '')
        if normalize_model_reference(name) not in skip_refs:
            result.append({**rec, **backend._download_fields(name, downloads)})
    return result


def _add_unlisted_models(
    backend: ElectronBackend,
    models: list[dict[str, Any]],
    downloads: dict[str, dict[str, Any]],
    current_model: str,
    models_dir: Path,
) -> None:
    from application.model_download import normalize_model_reference

    known = {
        normalize_model_reference(str(m.get('name') or '')) for m in models
    }
    if current_model:
        known = backend._add_if_missing(
            models, known, current_model, models_dir, downloads
        )
    for name in downloads:
        known = backend._add_if_missing(
            models, known, name, models_dir, downloads
        )


def _add_if_missing(
    backend: ElectronBackend,
    models: list[dict[str, Any]],
    known: set[str],
    name: str,
    models_dir: Path,
    downloads: dict[str, dict[str, Any]],
) -> set[str]:
    from application.model_download import normalize_model_reference

    ref = normalize_model_reference(name)
    if ref and ref not in known:
        models.append(backend._custom_model_record(name, models_dir, downloads))
        known.add(ref)
    return known


def _recommended_model_record(
    backend: ElectronBackend, name: str, models_dir: Path, downloads: dict
) -> dict[str, Any]:
    from application.model_download import is_builtin_model, is_model_cached
    from application.native_asr_models import (
        is_native_asr_model,
        native_asr_archive_url,
    )

    cached = is_model_cached(name, models_dir=models_dir)
    native = is_native_asr_model(name)
    status = 'missing_native' if native else 'recommended'
    if cached:
        status = 'compatible'
    return {
        'name': name,
        'label': name,
        'cached': cached,
        'compatible': cached,
        'status': status,
        'source': 'recommended',
        'builtin': is_builtin_model(name),
        'recommended': True,
        'downloadable': not native or native_asr_archive_url(name) is not None,
        'deletable': False,
        **backend._download_fields(name, downloads),
    }


def _custom_model_record(
    backend: ElectronBackend, name: str, models_dir: Path, downloads: dict
) -> dict[str, Any]:
    from application.model_download import is_model_cached

    cached = is_model_cached(name, models_dir=models_dir)
    return {
        'name': name,
        'label': name,
        'cached': cached,
        'compatible': cached,
        'status': 'compatible' if cached else 'unknown_remote',
        'source': 'custom',
        'builtin': False,
        'recommended': False,
        'downloadable': True,
        'deletable': False,
        **backend._download_fields(name, downloads),
    }


def download_model(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    from application.model_download import (
        download_model_async,
        normalize_model_reference,
    )
    from application.native_asr_models import (
        is_native_asr_model,
        native_asr_archive_url,
    )

    name = str(params.get('name') or '').strip()
    if not name:
        raise ValueError('download_model requires params.name')
    if is_native_asr_model(name) and native_asr_archive_url(name) is None:
        raise ValueError(
            'Native ASR models use their dedicated installer and cannot be downloaded as Faster Whisper models'
        )
    name = normalize_model_reference(name)
    models_dir = backend._models_dir_from_params(params)
    if 'useProxy' in params or 'use_proxy' in params:
        use_proxy = bool(
            first_param(params, 'useProxy', 'use_proxy', default=False)
        )
        proxy = str(params.get('proxy') or '').strip() if use_proxy else ''
    else:
        proxy = backend._model_download_proxy()
    with backend._download_lock:
        if backend._downloads.get(name, {}).get('state') == 'downloading':
            return {'started': False, 'message': 'Already downloading'}
        backend._downloads[name] = {'state': 'downloading'}
    backend._set_download_state(
        name,
        {
            'state': 'downloading',
            'message': 'Starting...',
            'error': '',
            'downloadedBytes': 0,
            'speedBps': 0,
            'proxy': bool(proxy),
        },
    )

    def on_progress(update: Any) -> None:
        payload = (
            update if isinstance(update, dict) else {'message': str(update)}
        )
        backend._set_download_state(
            name,
            {
                'state': 'downloading',
                'message': str(payload.get('message') or 'Downloading...'),
                'error': '',
                'downloadedBytes': int_or_zero(payload.get('downloadedBytes')),
                'speedBps': float(payload.get('speedBps') or 0.0),
                'proxy': bool(proxy),
            },
        )

    def on_done(error: str | None) -> None:
        if error:
            backend._set_download_state(
                name,
                {
                    'state': 'error',
                    'message': '',
                    'error': error,
                    'downloadedBytes': backend._download_record(name).get(
                        'downloadedBytes', 0
                    ),
                    'speedBps': 0,
                    'proxy': bool(proxy),
                },
            )
        else:
            backend._set_download_state(
                name,
                {
                    'state': 'done',
                    'message': 'Downloaded',
                    'error': '',
                    'downloadedBytes': backend._download_record(name).get(
                        'downloadedBytes', 0
                    ),
                    'speedBps': 0,
                    'proxy': bool(proxy),
                },
            )

    try:
        download_model_async(
            name, on_progress, on_done, models_dir=models_dir, proxy=proxy
        )
    except Exception as error:
        on_done(f'{type(error).__name__}: {error}')
        raise
    return {
        'started': True,
        'message': f'Downloading {name}...',
        'proxy': bool(proxy),
    }


def delete_model(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    from application.model_download import delete_local_model

    name = str(params.get('name') or '').strip()
    if backend._download_record(name).get('state') == 'downloading':
        raise RuntimeError(
            'Wait until the model download finishes before deleting it'
        )
    if backend._model_is_selected_or_running(name):
        raise RuntimeError(
            'Cannot delete the model that is selected or currently used by ASR'
        )
    delete_local_model(name, models_dir=backend._models_dir_from_params(params))
    backend._catalog_cache_clear()
    return {'deleted': True, 'name': name}


def model_metadata(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    from application.model_download import model_metadata

    name = str(params.get('name') or '').strip()
    if not name:
        raise ValueError('model_metadata requires params.name')
    return model_metadata(
        name, models_dir=backend._models_dir_from_params(params)
    )


def _asr_model_options(
    backend: ElectronBackend, config: dict[str, Any]
) -> list[str]:
    local = scan_compatible_asr_models(
        backend._models_dir_from_params(config=config)
    )
    ui = config.get('ui', {}) if isinstance(config.get('ui'), dict) else {}
    current = str(ui.get('model', '') or '').strip()
    seen: set[str] = set()
    out: list[str] = []
    for name in [*ASR_MODEL_NAMES, *local, current]:
        text = str(name or '').strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


class BackendAsrModelsMixin:
    """Compatibility facade for callers using the former mixin API."""

    list_models = list_models
    _build_model_list = _build_model_list
    _local_model_records = _local_model_records
    _add_unlisted_models = _add_unlisted_models
    _add_if_missing = _add_if_missing
    _recommended_model_record = _recommended_model_record
    _custom_model_record = _custom_model_record
    download_model = download_model
    delete_model = delete_model
    model_metadata = model_metadata
    _asr_model_options = _asr_model_options
