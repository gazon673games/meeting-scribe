from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from interface.backend_parts.system_utils import int_or_zero
from shared.domain.mappings import first_param

if TYPE_CHECKING:
    from interface.backend_impl import ElectronBackend


def list_diarization_models(
    backend: ElectronBackend, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    from application.diarization_model_download import (
        list_diarization_models,
    )

    models_dir = backend._models_dir_from_params(params, kind='diarization')
    downloads = backend._diarization_download_snapshot()
    cache_key = ('diar', str(models_dir))
    if backend._active_download_count(downloads) <= 0:
        cached = backend._catalog_cache_get(cache_key)
        if cached is not None:
            return cached
    result = list_diarization_models(
        project_root=backend.project_root,
        models_dir=models_dir,
        subdir='',
        downloads=downloads,
    )
    if backend._active_download_count(downloads) <= 0:
        backend._catalog_cache_put(cache_key, result)
    return result


def download_diarization_model(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    from application.diarization_model_download import (
        download_diarization_model_async,
    )

    name = str(params.get('name') or '').strip()
    if not name:
        raise ValueError('download_diarization_model requires params.name')
    if 'useProxy' in params or 'use_proxy' in params:
        use_proxy = bool(
            first_param(params, 'useProxy', 'use_proxy', default=False)
        )
        proxy = str(params.get('proxy') or '').strip() if use_proxy else ''
    else:
        proxy = backend._model_download_proxy()
    with backend._diarization_download_lock:
        if (
            backend._diarization_downloads.get(name, {}).get('state')
            == 'downloading'
        ):
            return {'started': False, 'message': 'Already downloading'}
        backend._diarization_downloads[name] = {'state': 'downloading'}
    backend._set_diarization_download_state(
        name,
        {
            'state': 'downloading',
            'message': 'Starting...',
            'error': '',
            'downloadedBytes': 0,
            'totalBytes': 0,
            'speedBps': 0,
            'proxy': bool(proxy),
        },
    )

    def on_progress(update: Any) -> None:
        payload = (
            update if isinstance(update, dict) else {'message': str(update)}
        )
        backend._set_diarization_download_state(
            name,
            {
                'state': 'downloading',
                'message': str(payload.get('message') or 'Downloading...'),
                'error': '',
                'downloadedBytes': int_or_zero(payload.get('downloadedBytes')),
                'totalBytes': int_or_zero(payload.get('totalBytes')),
                'speedBps': float(payload.get('speedBps') or 0.0),
                'path': str(payload.get('path') or ''),
                'proxy': bool(proxy),
            },
        )

    def on_done(error: str | None) -> None:
        previous = backend._diarization_download_record(name)
        backend._set_diarization_download_state(
            name,
            {
                'state': 'error' if error else 'done',
                'message': '' if error else 'Downloaded',
                'error': str(error or ''),
                'downloadedBytes': int_or_zero(previous.get('downloadedBytes')),
                'totalBytes': int_or_zero(previous.get('totalBytes')),
                'speedBps': 0,
                'path': str(previous.get('path') or ''),
                'proxy': bool(proxy),
            },
        )

    try:
        download_diarization_model_async(
            name=name,
            project_root=backend.project_root,
            models_dir=backend._models_dir_from_params(
                params, kind='diarization'
            ),
            subdir='',
            proxy=proxy,
            on_progress=on_progress,
            on_done=on_done,
        )
    except Exception as error:
        on_done(f'{type(error).__name__}: {error}')
        raise
    return {
        'started': True,
        'message': f'Downloading Speaker ID model {name}...',
        'proxy': bool(proxy),
    }


def delete_diarization_model(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    from application.diarization_model_download import (
        delete_diarization_model,
    )

    path = str(params.get('path') or '').strip()
    if not path:
        raise ValueError('delete_diarization_model requires params.path')
    wanted = Path(path).expanduser().resolve()
    records = backend.list_diarization_models(params).get('models', [])
    if any(
        record.get('downloading') and Path(record['path']).resolve() == wanted
        for record in records
    ):
        raise RuntimeError(
            'Wait until Speaker ID model downloads finish before deleting models'
        )
    if backend._diarization_model_is_selected_or_running(path):
        raise RuntimeError(
            'Cannot delete the Speaker ID model that is selected or currently used by ASR'
        )
    delete_diarization_model(
        project_root=backend.project_root,
        path=path,
        models_dir=backend._models_dir_from_params(params, kind='diarization'),
        subdir='',
    )
    backend._catalog_cache_clear()
    return {'deleted': True, 'path': path}


def list_llm_models(
    backend: ElectronBackend, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    from application.llm_model_download import list_llm_models

    models_dir = backend._llm_models_dir_from_params(params)
    downloads = backend._llm_download_snapshot()
    cache_key = ('llm', str(models_dir))
    if backend._active_download_count(downloads) <= 0:
        cached = backend._catalog_cache_get(cache_key)
        if cached is not None:
            return cached
    result = list_llm_models(
        project_root=backend.project_root,
        models_dir=models_dir,
        downloads=downloads,
    )
    if backend._active_download_count(downloads) <= 0:
        backend._catalog_cache_put(cache_key, result)
    return result


def download_llm_model(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    from application.llm_model_download import (
        download_llm_model_async,
        parse_llm_source,
    )

    name = str(params.get('name') or '').strip()
    if not name:
        raise ValueError('download_llm_model requires params.name')
    source = parse_llm_source(name)
    key = source.filename or source.folder or source.repo_id or name
    if 'useProxy' in params or 'use_proxy' in params:
        use_proxy = bool(
            first_param(params, 'useProxy', 'use_proxy', default=False)
        )
        proxy = str(params.get('proxy') or '').strip() if use_proxy else ''
    else:
        proxy = backend._model_download_proxy()
    with backend._llm_download_lock:
        if backend._llm_downloads.get(key, {}).get('state') == 'downloading':
            return {'started': False, 'message': 'Already downloading'}
        backend._llm_downloads[key] = {'state': 'downloading'}
    backend._set_llm_download_state(
        key,
        {
            'state': 'downloading',
            'message': 'Starting...',
            'error': '',
            'downloadedBytes': 0,
            'totalBytes': 0,
            'speedBps': 0,
            'path': '',
            'proxy': bool(proxy),
        },
    )

    def on_progress(update: Any) -> None:
        payload = (
            update if isinstance(update, dict) else {'message': str(update)}
        )
        backend._set_llm_download_state(
            key,
            {
                'state': 'downloading',
                'message': str(payload.get('message') or 'Downloading...'),
                'error': '',
                'downloadedBytes': int_or_zero(payload.get('downloadedBytes')),
                'totalBytes': int_or_zero(payload.get('totalBytes')),
                'speedBps': float(payload.get('speedBps') or 0.0),
                'path': str(payload.get('path') or ''),
                'proxy': bool(proxy),
            },
        )

    def on_done(error: str | None) -> None:
        previous = backend._llm_download_record(key)
        backend._set_llm_download_state(
            key,
            {
                'state': 'error' if error else 'done',
                'message': '' if error else 'Downloaded',
                'error': str(error or ''),
                'downloadedBytes': int_or_zero(previous.get('downloadedBytes')),
                'totalBytes': int_or_zero(previous.get('totalBytes')),
                'speedBps': 0,
                'path': str(previous.get('path') or ''),
                'proxy': bool(proxy),
            },
        )

    try:
        download_llm_model_async(
            name=name,
            project_root=backend.project_root,
            models_dir=backend._llm_models_dir_from_params(params),
            proxy=proxy,
            on_progress=on_progress,
            on_done=on_done,
        )
    except Exception as error:
        on_done(f'{type(error).__name__}: {error}')
        raise
    label = source.filename or source.repo_id or name
    return {
        'started': True,
        'message': f'Downloading language model {label}...',
        'proxy': bool(proxy),
    }


def delete_llm_model(
    backend: ElectronBackend, params: dict[str, Any]
) -> dict[str, Any]:
    from application.llm_model_download import delete_llm_model

    path = str(params.get('path') or '').strip()
    if not path:
        raise ValueError('delete_llm_model requires params.path')
    wanted = Path(path).expanduser().resolve()
    downloads = backend._llm_download_snapshot()
    if any(
        record.get('state') == 'downloading'
        and (
            Path(record['path']).resolve() == wanted
            if record.get('path')
            else name in (wanted.name, wanted.parent.name)
        )
        for name, record in downloads.items()
    ):
        raise RuntimeError(
            'Wait until LLM downloads finish before deleting models'
        )
    from infrastructure.llama_cpp_runner import model_path_in_use as gguf_in_use
    from infrastructure.local_llm_parts.runtime_server import (
        model_path_in_use as server_in_use,
    )

    if gguf_in_use(Path(path)) or server_in_use(Path(path)):
        raise RuntimeError('Stop the local model before deleting its file')
    delete_llm_model(
        project_root=backend.project_root,
        path=path,
        models_dir=backend._llm_models_dir_from_params(params),
    )
    backend._catalog_cache_clear()
    return {'deleted': True, 'path': path}


class BackendExtraModelsMixin:
    """Compatibility facade for callers using the former mixin API."""

    list_diarization_models = list_diarization_models
    download_diarization_model = download_diarization_model
    delete_diarization_model = delete_diarization_model
    list_llm_models = list_llm_models
    download_llm_model = download_llm_model
    delete_llm_model = delete_llm_model
