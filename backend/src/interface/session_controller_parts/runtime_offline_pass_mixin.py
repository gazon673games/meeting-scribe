from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from application.asr_language import runtime_asr_language
from application.session_tasks import OfflinePassRequest
from shared.domain.mappings import first_param

if TYPE_CHECKING:
    from interface.session_controller import HeadlessSessionController


def _should_run_offline_pass(
    session: HeadlessSessionController, params: dict[str, Any], wav_path: Path
) -> bool:
    return bool(
        params.get('runOfflinePass', False)
        and session.offline_pass_use_case is not None
        and session.offline_pass_use_case.available()
        and Path(wav_path).is_file()
    )


def _start_offline_pass(
    session: HeadlessSessionController, wav_path: Path, params: dict[str, Any]
) -> None:
    if session.offline_pass_use_case is None:
        return
    with session._lock:
        if session._closing or session._offline_pass_running:
            return
        session._session_state.begin_offline_pass(str(wav_path))
        session._offline_pass_running = True
        session._offline_pass_result = {
            'wavPath': str(wav_path),
            'status': 'running',
            'ts': time.time(),
        }
        session._offline_pass_thread = threading.Thread(
            target=session._run_offline_pass,
            args=(Path(wav_path), dict(params)),
            name='electron-offline-pass',
            daemon=True,
        )
        try:
            session._emit(
                'offline_pass_started', dict(session._offline_pass_result)
            )
            session._offline_pass_thread.start()
        except BaseException as error:
            session._offline_pass_thread = None
            session._offline_pass_running = False
            session._offline_pass_result.update(
                status='error', error=str(error)
            )
            session._session_state.finish_offline_pass(error=str(error))
            raise


def _run_offline_pass(
    session: HeadlessSessionController, wav_path: Path, params: dict[str, Any]
) -> None:
    assert session.offline_pass_use_case is not None
    error = ''
    out_txt = ''
    try:
        result = session.offline_pass_use_case.execute(
            OfflinePassRequest(
                project_root=session.project_root,
                models_dir=Path(params['asrModelsDir'])
                if params.get('asrModelsDir')
                else None,
                wav_path=Path(wav_path),
                model_name=str(
                    first_param(
                        params,
                        'offlineModelName',
                        'model',
                        default='large-v3',
                    )
                    or 'large-v3'
                ),
                language=runtime_asr_language(
                    str(params.get('language', 'ru'))
                ),
            )
        )
        out_txt = str(result.out_txt)
        payload = {
            'status': 'done',
            'wavPath': str(wav_path),
            'outTxt': out_txt,
            'ts': time.time(),
        }
    except Exception as exception:
        error = f'{type(exception).__name__}: {exception}'
        payload = {
            'status': 'error',
            'wavPath': str(wav_path),
            'error': error,
            'ts': time.time(),
        }
    with session._lock:
        session._offline_pass_running = False
        session._offline_pass_result = {
            'wavPath': str(wav_path),
            'status': 'error' if error else 'done',
            'error': error,
            'outTxt': out_txt,
            'ts': time.time(),
        }
        session._session_state.finish_offline_pass(out_txt=out_txt, error=error)
        if error:
            session._last_error = error
    session._emit(
        'offline_pass_error' if error else 'offline_pass_done', payload
    )


class RuntimeOfflinePassMixin:
    """Compatibility facade for callers using the former mixin API."""

    _should_run_offline_pass = _should_run_offline_pass
    _start_offline_pass = _start_offline_pass
    _run_offline_pass = _run_offline_pass
