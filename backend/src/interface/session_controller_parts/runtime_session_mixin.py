from __future__ import annotations

import logging
import threading
import time
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from application.local_paths import project_recordings_dir
from interface.session_controller_parts.helpers import (
    drops_record,
    line_speaker_or_stream,
    master_record,
    wav_requested,
)
from session.domain.state import SessionState
from shared.domain.mappings import first_param

_logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from interface.session_controller import HeadlessSessionController


def start_session(
    session: HeadlessSessionController, params: dict[str, Any]
) -> dict[str, Any]:
    with session._lock:
        asr_enabled = session._prepare_start_locked(params)
        try:
            wav_error, asr_error = session._start_runtime_features_locked(
                params, asr_enabled
            )
            session._configure_transcript_files_locked(params)
            return session._finalize_start_locked(
                wav_error=wav_error, asr_error=asr_error
            )
        except BaseException as error:
            try:
                session._stop_runtime_locked({})
            except Exception:
                _logger.exception('Cannot finish session startup rollback')
            session._last_error = f'{type(error).__name__}: {error}'
            if session._session_state.state == SessionState.STARTING:
                session._session_state.fail_start(session._last_error)
            elif session._session_state.can_stop:
                session._session_state.begin_stop(run_offline_pass=False)
                session._session_state.finish_stop(session._last_error)
            raise


def stop_session(
    session: HeadlessSessionController, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    failure: BaseException | None = None
    payload: dict[str, Any] = {}
    with session._lock:
        stop_params = session._prepare_stop_params_locked(params)
        wav_path = Path(str(stop_params.get('wavPath', '')))
        try:
            session._stop_runtime_locked(stop_params)
            payload = session._finalize_stop_locked(stop_params, wav_path)
        except BaseException as error:
            failure = error
            session._last_error = str(error)
            try:
                session._close_transcript_files_locked()
            except Exception:
                _logger.exception(
                    'Cannot close transcript files after stop failure'
                )
            if session._session_state.is_stopping:
                session._session_state.finish_stop(str(error))
        pump = session._asr_event_thread
    if pump is not None and pump is not threading.current_thread():
        pump.join(timeout=2)
        if pump.is_alive():
            if failure is not None:
                _logger.error(
                    'ASR event pump did not stop after session failure'
                )
                raise failure
            raise TimeoutError('ASR event pump did not stop')
    if failure is not None:
        raise failure
    return payload


def snapshot(session: HeadlessSessionController) -> dict[str, Any]:
    with session._lock:
        meters = session._safe_meters()
        settings = session._active_asr_settings
        return {
            'status': session._status,
            'running': bool(session._session_state.is_running),
            'state': session._session_state.state.value,
            'asrRunning': bool(session._asr_running),
            'asrRequested': bool(session._asr_requested),
            'asrModel': settings.model_name if settings is not None else '',
            'diarizationModelPath': (
                settings.diar_sherpa_embedding_model_path
                if settings is not None
                else ''
            ),
            'wavRecording': bool(session.writer.is_recording()),
            'wavPath': str(session.writer.target_path() or ''),
            'lastError': session._last_error,
            'lastWarning': session._last_warning,
            'master': master_record(meters),
            'drops': drops_record(meters, session.writer),
            'asrMetrics': dict(session._asr_metrics),
            'offlinePass': {
                'running': bool(session._offline_pass_running),
                'result': dict(session._offline_pass_result),
            },
            'humanLogPath': str(session._human_log_path or ''),
            'realtimeTranscriptPath': str(
                session._realtime_transcript_path or ''
            ),
            'transcript': deepcopy(session._transcript_lines[-200:]),
            'sources': [
                session._source_record(name, meters=meters)
                for name in session._sources
            ],
            'modelDownload': dict(session._model_download_info),
        }


def transcript_text(
    session: HeadlessSessionController, *, limit: int = 500
) -> str:
    limit = max(0, int(limit))
    if limit == 0:
        return ''
    with session._lock:
        lines = list(session._transcript_lines[-limit:])
    return '\n'.join(
        f'[{time.strftime("%H:%M:%S", time.localtime(float(line.get("ts", time.time()))))}] '
        f'{line_speaker_or_stream(line)}: {line.get("text", "")}'
        for line in lines
    )


def shutdown(session: HeadlessSessionController) -> None:
    with session._lock:
        if session._shutdown_complete:
            return
        session._closing = True
    failures: list[BaseException] = []
    try:
        if session._session_state.can_stop or session._asr_runtime is not None:
            session.stop_session({'runOfflinePass': False})
        if session._asr_runtime is not None:
            raise RuntimeError(
                session._last_error or 'ASR runtime did not stop'
            )
    except BaseException as error:
        failures.append(error)
        _logger.exception('Cannot stop session during shutdown')
    try:
        session.writer.stop()
    except BaseException as error:
        failures.append(error)
        _logger.exception('Cannot stop WAV writer during shutdown')
    with session._lock:
        try:
            session._close_transcript_files_locked()
        except BaseException as error:
            failures.append(error)
            _logger.exception('Cannot close transcript files during shutdown')
        offline_worker = session._offline_pass_thread
    if (
        offline_worker is not None
        and offline_worker is not threading.current_thread()
    ):
        offline_worker.join()
    if failures:
        raise failures[0]
    with session._lock:
        session._shutdown_complete = True


def _prepare_start_locked(
    session: HeadlessSessionController, params: dict[str, Any]
) -> bool:
    if session._closing:
        raise RuntimeError('Session is shutting down')
    if session._asr_runtime is not None:
        raise RuntimeError(
            'Previous ASR runtime needs cleanup; retry stopping it'
        )
    if not session._session_state.can_start:
        raise RuntimeError(
            f'Session is busy: {session._session_state.state.value}'
        )
    if not session._sources:
        raise RuntimeError('Add at least one source before starting a session')

    asr_enabled = bool(
        first_param(params, 'asrEnabled', 'asr_enabled', default=False)
    )
    session._asr_requested = bool(asr_enabled)
    session._last_warning = ''
    session._last_error = ''
    session._session_state.begin_start(
        source_count=len(session._sources),
        asr_enabled=asr_enabled,
        profile=str(params.get('profile', '')),
        language=str(params.get('language', '')),
    )

    try:
        for source in session._sources.values():
            session.engine.set_source_enabled(source.name, source.enabled)
            session.engine.set_source_delay_ms(source.name, source.delay_ms)
        if asr_enabled:
            session._drain_tap_queue()
        session._drain_asr_events()
        session._session_transcript_lines.clear()
        session._pending_speaker_updates.clear()
        session._apply_tap_config()
        session._set_engine_output_enabled(wav_requested(params))
        session.start()
        session.engine.start()
        return asr_enabled
    except Exception as error:
        try:
            session.engine.stop()
        except Exception:
            _logger.exception('Cannot stop audio engine after failed startup')
        session._last_error = f'{type(error).__name__}: {error}'
        session._session_state.fail_start(session._last_error)
        session._status = 'start failed'
        session._emit('session_error', {'message': session._last_error})
        raise


def _start_runtime_features_locked(
    session: HeadlessSessionController,
    params: dict[str, Any],
    asr_enabled: bool,
) -> tuple[str, str]:
    wav_error = session._start_wav_if_requested(params)
    if wav_error:
        session._set_engine_output_enabled(False)
    asr_error = session._start_asr_locked(params) if asr_enabled else ''
    return wav_error, asr_error


def _finalize_start_locked(
    session: HeadlessSessionController, *, wav_error: str, asr_error: str
) -> dict[str, Any]:
    session._session_state.finish_start(
        asr_running=session._asr_running,
        wav_recording=session.writer.is_recording(),
    )
    session._status = 'running'
    payload = session.snapshot()
    warnings = [
        warning
        for warning in (wav_error, asr_error, session._last_warning)
        if warning
    ]
    if warnings:
        payload['warning'] = ' '.join(warnings)
    session._emit('session_started', payload)
    return payload


def _prepare_stop_params_locked(
    session: HeadlessSessionController, params: dict[str, Any] | None
) -> dict[str, Any]:
    stop_params = dict(params or {})
    if session._session_state.is_stopping:
        raise RuntimeError('Session is already stopping')
    if not session._session_state.can_stop:
        if session._asr_runtime is not None:
            stop_params.setdefault(
                'wavPath', str(session._resolve_stop_wav_path(stop_params))
            )
            return stop_params
        if (
            not session.engine.is_running()
            and not session.writer.is_recording()
        ):
            return {'_snapshotOnly': True}
        raise RuntimeError(
            f'Cannot stop while session is {session._session_state.state.value}'
        )

    wav_path = session.writer.target_path() or session._current_output_path(
        str(stop_params.get('outputFile', ''))
    )
    stop_params.setdefault('wavPath', str(wav_path))
    session._session_state.begin_stop(
        run_offline_pass=bool(stop_params.get('runOfflinePass', False))
    )
    return stop_params


def _stop_runtime_locked(
    session: HeadlessSessionController, stop_params: dict[str, Any]
) -> None:
    if stop_params.get('_snapshotOnly'):
        return

    def stop_recording() -> None:
        if session.writer.is_recording():
            session.writer.stop_recording()

    def stop_engine() -> None:
        if session.engine.is_running():
            session.engine.stop()

    failures: list[BaseException] = []
    for operation in (
        stop_recording,
        partial(session._set_engine_output_enabled, False),
        partial(session._stop_asr_locked, stop_params),
        stop_engine,
        partial(session.engine.set_tap_queue, None),
        session._drain_tap_queue,
    ):
        try:
            operation()
        except BaseException as error:
            failures.append(error)
            _logger.exception('Session shutdown operation failed')
    if failures:
        raise failures[0]


def _finalize_stop_locked(
    session: HeadlessSessionController,
    stop_params: dict[str, Any],
    wav_path: Path,
) -> dict[str, Any]:
    if stop_params.get('_snapshotOnly'):
        return session.snapshot()

    session._asr_requested = False
    session._asr_running = session._asr_runtime is not None
    session._close_transcript_files_locked()
    if session._session_state.is_stopping:
        session._session_state.finish_stop(
            session._last_error if session._asr_running else ''
        )
    session._status = 'idle'
    if not session._asr_running and session._should_run_offline_pass(
        stop_params, wav_path
    ):
        session._start_offline_pass(wav_path, stop_params)
    payload = session.snapshot()
    session._emit('session_stopped', payload)
    return payload


def _start_wav_if_requested(
    session: HeadlessSessionController, params: dict[str, Any]
) -> str:
    if not wav_requested(params):
        return ''
    if not session.wav_recorder_factory.available():
        return (
            'WAV recording is unavailable because soundfile is not installed.'
        )
    output_file = str(
        first_param(params, 'outputFile', 'output_file', default='') or ''
    ).strip()
    output_path = session._current_output_path(output_file)
    try:
        session.writer.start_recording(output_path, session.format)
    except Exception as error:
        return f'WAV recording could not start: {type(error).__name__}: {error}'
    return ''


def _set_engine_output_enabled(
    session: HeadlessSessionController, enabled: bool
) -> None:
    set_output_enabled = getattr(session.engine, 'set_output_enabled', None)
    if callable(set_output_enabled):
        try:
            set_output_enabled(bool(enabled))
        except Exception:
            _logger.debug(
                'Best-effort operation failed in _set_engine_output_enabled',
                exc_info=True,
            )


def _current_output_path(
    session: HeadlessSessionController, raw_name: str
) -> Path:
    name = Path(raw_name or 'capture_mix.wav').name
    if not name.lower().endswith('.wav'):
        name += '.wav'
    return project_recordings_dir(session.project_root, create=True) / name


def _apply_tap_config(session: HeadlessSessionController) -> None:
    if not session._asr_requested:
        session.engine.set_tap_queue(None)
        return
    session.engine.set_tap_queue(session.tap_queue)
    session.engine.set_tap_config(
        mode='both',
        sources=[
            source.name
            for source in session._sources.values()
            if source.enabled
        ],
        drop_threshold=0.85,
    )


class RuntimeSessionMixin:
    """Compatibility facade for callers using the former mixin API."""

    start_session = start_session
    stop_session = stop_session
    snapshot = snapshot
    transcript_text = transcript_text
    shutdown = shutdown
    _prepare_start_locked = _prepare_start_locked
    _start_runtime_features_locked = _start_runtime_features_locked
    _finalize_start_locked = _finalize_start_locked
    _prepare_stop_params_locked = _prepare_stop_params_locked
    _stop_runtime_locked = _stop_runtime_locked
    _finalize_stop_locked = _finalize_stop_locked
    _start_wav_if_requested = _start_wav_if_requested
    _set_engine_output_enabled = _set_engine_output_enabled
    _current_output_path = _current_output_path
    _apply_tap_config = _apply_tap_config
