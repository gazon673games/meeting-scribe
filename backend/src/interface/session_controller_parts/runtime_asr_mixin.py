from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from application.asr_language import runtime_asr_language
from application.asr_session import ASRRuntime, ASRSessionSettings
from application.session_tasks import StopAsrRequest
from interface.session_controller_parts.artifacts import (
    generated_session_id,
    write_session_outputs,
)
from interface.session_controller_parts.settings import asr_settings_from_params
from shared.domain.mappings import first_param

if TYPE_CHECKING:
    from interface.session_controller import HeadlessSessionController


def _write_session_outputs_locked(
    session: HeadlessSessionController,
    *,
    asr: ASRRuntime,
    settings: ASRSessionSettings | None,
    session_id: str,
) -> None:
    write_session_outputs(
        asr=asr,
        settings=settings,
        session_id=session_id,
        project_root=session.project_root,
        transcript_lines=session._session_transcript_lines,
    )


def _start_asr_locked(
    session: HeadlessSessionController, params: dict[str, Any]
) -> str:
    if (
        session.asr_runtime_factory is None
        or session.transcription_startup_service is None
    ):
        session._last_warning = (
            'Live ASR is not configured for the Electron backend yet.'
        )
        return session._last_warning

    session._drain_asr_events()
    settings = asr_settings_from_params(
        params, source_speaker_labels=session._source_speaker_labels_locked()
    )
    session._active_asr_settings = settings
    session._transcription_state.begin_start(
        model_name=settings.model_name,
        mode=str(settings.mode),
        language=settings.language,
    )
    result = session.transcription_startup_service.start(
        settings,
        runtime_factory=session.asr_runtime_factory,
        tap_queue=session.tap_queue,
        project_root=session.project_root,
        event_queue=session.asr_event_queue,
    )

    for error in result.errors:
        session._emit('asr_start_attempt_failed', {'message': error})

    if not result.ok or result.asr is None or result.attempt is None:
        message = '; '.join(result.errors) or 'unknown ASR startup error'
        session._transcription_state.fail_start(message)
        session._asr_running = False
        session._asr_runtime = None
        session._active_asr_session_id = ''
        session._last_warning = f'Live ASR could not start: {message}'
        return session._last_warning

    session._asr_runtime = result.asr
    session._asr_running = True
    session._active_asr_session_id = str(
        getattr(result.asr, 'session_id', '') or generated_session_id()
    )
    session._transcription_state.finish_start(
        degraded=result.attempt.degraded, attempt_label=result.attempt.label
    )
    session._start_asr_event_pump()
    if result.degraded:
        session._last_warning = (
            f'Live ASR started in fallback mode: {result.attempt.label}.'
        )
        return session._last_warning
    session._last_warning = ''
    return ''


def _resolve_stop_wav_path(
    session: HeadlessSessionController, params: dict[str, Any]
) -> Path:
    return Path(
        str(
            params.get('wavPath')
            or session.writer.target_path()
            or session._current_output_path(str(params.get('outputFile', '')))
        )
    )


def _build_stop_asr_request(
    session: HeadlessSessionController,
    asr: ASRRuntime,
    wav_path: Path,
    params: dict[str, Any],
) -> StopAsrRequest:
    p = params or {}
    return StopAsrRequest(
        asr=asr,
        wav_path=wav_path,
        run_offline_pass=bool(p.get('runOfflinePass', False)),
        offline_model_name=str(
            first_param(p, 'offlineModelName', 'model', default='large-v3')
        ),
        offline_language=runtime_asr_language(str(p.get('language', 'ru'))),
    )


def _stop_asr_locked(
    session: HeadlessSessionController, params: dict[str, Any]
) -> None:
    asr = session._asr_runtime
    session._asr_event_stop.set()
    session._transcript_lines = [
        line
        for line in session._transcript_lines
        if not str(line.get('id', '')).startswith('streaming-')
    ]
    if asr is None:
        session._asr_running = False
        session._transcription_state.reset()
        session._active_asr_settings = None
        session._active_asr_session_id = ''
        return

    if session._transcription_state.can_stop:
        session._transcription_state.begin_stop()
    wav_path = session._resolve_stop_wav_path(params)
    result = session.stop_asr_use_case.execute(
        session._build_stop_asr_request(asr, wav_path, params)
    )
    stop_error = str(result.stop_error or '')
    if session._transcription_state.is_stopping:
        session._transcription_state.finish_stop(stop_error)
    else:
        session._transcription_state.reset()
    if stop_error:
        session._asr_running = True
        session._last_error = stop_error
        session._emit('asr_stop_error', {'message': stop_error})
        return
    session._asr_runtime = None
    session._asr_running = False
    try:
        session._drain_asr_events()
        session._write_session_outputs_locked(
            asr=asr,
            settings=session._active_asr_settings,
            session_id=session._active_asr_session_id,
        )
    finally:
        session._active_asr_settings = None
        session._active_asr_session_id = ''


def _start_asr_event_pump(session: HeadlessSessionController) -> None:
    session._asr_event_stop = threading.Event()
    worker = threading.Thread(
        target=session._run_asr_event_pump,
        args=(session._asr_event_stop,),
        name='electron-asr-events',
        daemon=True,
    )
    try:
        worker.start()
    except BaseException:
        session._asr_event_stop.set()
        raise
    session._asr_event_thread = worker


def _run_asr_event_pump(
    session: HeadlessSessionController, stop: threading.Event | None = None
) -> None:
    if stop is None:
        stop = session._asr_event_stop
    while not stop.is_set():
        with session._lock:
            if stop.is_set():
                return
            try:
                item = session.asr_event_queue.get_nowait()
            except queue.Empty:
                item = None
            if item is not None:
                session._handle_asr_event(item)
                continue
        stop.wait(0.02)


def _drain_asr_events(
    session: HeadlessSessionController, *, limit: int | None = None
) -> int:
    count = 0
    while limit is None or count < limit:
        try:
            item = session.asr_event_queue.get_nowait()
        except queue.Empty:
            break
        session._handle_asr_event(item)
        count += 1
    return count


def _drain_tap_queue(session: HeadlessSessionController) -> int:
    count = 0
    while True:
        try:
            session.tap_queue.get_nowait()
            count += 1
        except queue.Empty:
            return count


class RuntimeAsrMixin:
    """Compatibility facade for callers using the former mixin API."""

    _write_session_outputs_locked = _write_session_outputs_locked
    _start_asr_locked = _start_asr_locked
    _resolve_stop_wav_path = _resolve_stop_wav_path
    _build_stop_asr_request = _build_stop_asr_request
    _stop_asr_locked = _stop_asr_locked
    _start_asr_event_pump = _start_asr_event_pump
    _run_asr_event_pump = _run_asr_event_pump
    _drain_asr_events = _drain_asr_events
    _drain_tap_queue = _drain_tap_queue
