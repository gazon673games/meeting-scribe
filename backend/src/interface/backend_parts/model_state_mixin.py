from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from interface.backend_parts.records import DownloadState
from interface.backend_parts.system_utils import int_or_zero

if TYPE_CHECKING:
    from interface.backend_impl import ElectronBackend


def _download_snapshot(backend: ElectronBackend) -> dict[str, DownloadState]:
    with backend._download_lock:
        return {
            str(name): record.copy()
            for name, record in backend._downloads.items()
        }


def _diarization_download_snapshot(
    backend: ElectronBackend,
) -> dict[str, DownloadState]:
    with backend._diarization_download_lock:
        return {
            str(name): record.copy()
            for name, record in backend._diarization_downloads.items()
        }


def _llm_download_snapshot(
    backend: ElectronBackend,
) -> dict[str, DownloadState]:
    with backend._llm_download_lock:
        return {
            str(name): record.copy()
            for name, record in backend._llm_downloads.items()
        }


def _download_record(backend: ElectronBackend, name: str) -> dict[str, Any]:
    from application.model_download import normalize_model_reference

    key = normalize_model_reference(name)
    with backend._download_lock:
        return dict(
            backend._downloads.get(name) or backend._downloads.get(key) or {}
        )


def _diarization_download_record(
    backend: ElectronBackend, name: str
) -> dict[str, Any]:
    with backend._diarization_download_lock:
        return dict(backend._diarization_downloads.get(str(name)) or {})


def _llm_download_record(backend: ElectronBackend, name: str) -> dict[str, Any]:
    with backend._llm_download_lock:
        return dict(backend._llm_downloads.get(str(name)) or {})


def _download_fields(
    backend: ElectronBackend, name: str, downloads: dict[str, DownloadState]
) -> dict[str, Any]:
    from application.model_download import normalize_model_reference

    key = normalize_model_reference(name)
    dl = downloads.get(name) or downloads.get(key) or {}
    return {
        'downloading': dl.get('state') == 'downloading',
        'downloadDone': dl.get('state') == 'done',
        'downloadError': str(dl.get('error') or ''),
        'downloadMessage': str(dl.get('message') or ''),
        'downloadedBytes': int_or_zero(dl.get('downloadedBytes')),
        'speedBps': float(dl.get('speedBps') or 0.0),
        'downloadUsesProxy': bool(dl.get('proxy', False)),
    }


def _catalog_cache_get(
    backend: ElectronBackend, key: tuple[Any, ...], *, max_age_s: float = 10.0
) -> dict[str, Any] | None:
    now = time.monotonic()
    with backend._catalog_cache_lock:
        cached = backend._catalog_cache.get(tuple(key))
        if cached is None:
            return None
        ts, value = cached
        if now - float(ts) > float(max_age_s):
            backend._catalog_cache.pop(tuple(key), None)
            return None
        return copy.deepcopy(value)


def _catalog_cache_put(
    backend: ElectronBackend, key: tuple[Any, ...], value: dict[str, Any]
) -> None:
    with backend._catalog_cache_lock:
        backend._catalog_cache[tuple(key)] = (
            time.monotonic(),
            copy.deepcopy(value),
        )


def _catalog_cache_clear(backend: ElectronBackend) -> None:
    with backend._catalog_cache_lock:
        backend._catalog_cache.clear()


def _set_download_state(
    backend: ElectronBackend, name: str, record: DownloadState
) -> None:
    with backend._download_lock:
        backend._downloads[str(name)] = record.copy()
        active_downloads = backend._active_download_count(backend._downloads)
    backend._catalog_cache_clear()
    backend._emit(
        'model_download_updated',
        {
            'model': str(name),
            **dict(record),
            'activeDownloads': active_downloads,
        },
    )


def _set_diarization_download_state(
    backend: ElectronBackend, name: str, record: DownloadState
) -> None:
    with backend._diarization_download_lock:
        backend._diarization_downloads[str(name)] = record.copy()
        active_downloads = backend._active_download_count(
            backend._diarization_downloads
        )
    backend._catalog_cache_clear()
    backend._emit(
        'diarization_model_download_updated',
        {
            'model': str(name),
            **dict(record),
            'activeDownloads': active_downloads,
        },
    )


def _set_llm_download_state(
    backend: ElectronBackend, name: str, record: DownloadState
) -> None:
    with backend._llm_download_lock:
        backend._llm_downloads[str(name)] = record.copy()
        active_downloads = backend._active_download_count(
            backend._llm_downloads
        )
    backend._catalog_cache_clear()
    backend._emit(
        'llm_model_download_updated',
        {
            'model': str(name),
            **dict(record),
            'activeDownloads': active_downloads,
        },
    )


def _active_download_count(
    backend: ElectronBackend, downloads: dict[str, DownloadState] | None = None
) -> int:
    source = downloads
    if source is None:
        with backend._download_lock:
            source = dict(backend._downloads)
    return sum(
        1 for record in source.values() if record.get('state') == 'downloading'
    )


def _model_download_proxy(
    backend: ElectronBackend, config: dict[str, Any] | None = None
) -> str:
    cfg = config if isinstance(config, dict) else backend._read_config()
    models = (
        cfg.get('models', {}) if isinstance(cfg.get('models'), dict) else {}
    )
    if not bool(models.get('use_proxy', False)):
        return ''
    model_proxy = str(models.get('proxy') or '').strip()
    if model_proxy:
        return model_proxy
    codex = cfg.get('codex', {}) if isinstance(cfg.get('codex'), dict) else {}
    return str(codex.get('proxy') or '').strip()


def _model_is_selected_or_running(backend: ElectronBackend, name: str) -> bool:
    from application.model_download import normalize_model_reference

    wanted = normalize_model_reference(name)
    config = backend._read_config()
    current = str(
        (
            config.get('ui', {}) if isinstance(config.get('ui'), dict) else {}
        ).get('model', '')
        or ''
    ).strip()
    if wanted and normalize_model_reference(current) == wanted:
        return True
    session = backend._session_snapshot()
    active_model = str(session.get('asrModel') or '')
    return bool(wanted and normalize_model_reference(active_model) == wanted)


def _diarization_model_is_selected_or_running(
    backend: ElectronBackend, path: str
) -> bool:
    wanted = str(Path(path).expanduser().resolve())
    config = backend._read_config()
    asr = config.get('asr', {}) if isinstance(config.get('asr'), dict) else {}
    current = str(asr.get('diar_sherpa_embedding_model_path') or '').strip()
    if current and str(Path(current).expanduser().resolve()) == wanted:
        return True
    session = backend._session_snapshot()
    active_path = str(session.get('diarizationModelPath') or '')
    return bool(
        active_path and str(Path(active_path).expanduser().resolve()) == wanted
    )


class BackendModelStateMixin:
    """Compatibility facade for callers using the former mixin API."""

    _download_snapshot = _download_snapshot
    _diarization_download_snapshot = _diarization_download_snapshot
    _llm_download_snapshot = _llm_download_snapshot
    _download_record = _download_record
    _diarization_download_record = _diarization_download_record
    _llm_download_record = _llm_download_record
    _download_fields = _download_fields
    _catalog_cache_get = _catalog_cache_get
    _catalog_cache_put = _catalog_cache_put
    _catalog_cache_clear = _catalog_cache_clear
    _set_download_state = _set_download_state
    _set_diarization_download_state = _set_diarization_download_state
    _set_llm_download_state = _set_llm_download_state
    _active_download_count = _active_download_count
    _model_download_proxy = _model_download_proxy
    _model_is_selected_or_running = _model_is_selected_or_running
    _diarization_model_is_selected_or_running = (
        _diarization_model_is_selected_or_running
    )
