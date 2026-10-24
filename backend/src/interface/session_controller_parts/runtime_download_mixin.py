from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from interface.session_controller import HeadlessSessionController


def begin_model_download(
    session: HeadlessSessionController, model_name: str
) -> None:
    with session._lock:
        session._session_state.begin_model_download(model_name)
        session._model_download_info = {
            'model': model_name,
            'downloadedBytes': 0,
            'speedBps': 0.0,
            'message': 'Starting download...',
        }
        session._emit(
            'session_state_changed',
            {'state': 'downloading_model', 'model': model_name},
        )


def update_model_download_progress(
    session: HeadlessSessionController, info: dict[str, Any]
) -> None:
    with session._lock:
        session._model_download_info = {
            'model': session._model_download_info.get('model', ''),
            'downloadedBytes': int(info.get('downloadedBytes', 0)),
            'speedBps': float(info.get('speedBps', 0.0)),
            'message': str(info.get('message', '')),
        }


def finish_model_download(
    session: HeadlessSessionController, error: str = ''
) -> None:
    with session._lock:
        model_name = session._model_download_info.get('model', '')
        session._session_state.finish_model_download(
            model_name=model_name, error=error
        )
        session._model_download_info = {}
        if error:
            session._last_error = error
            session._emit('session_error', {'message': error})
        session._emit(
            'session_state_changed', {'state': 'idle', 'error': error}
        )


class RuntimeDownloadMixin:
    """Compatibility facade for callers using the former mixin API."""

    begin_model_download = begin_model_download
    update_model_download_progress = update_model_download_progress
    finish_model_download = finish_model_download
