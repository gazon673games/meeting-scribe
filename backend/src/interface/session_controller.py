from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np

from application.asr_session import (
    ASRRuntime,
    ASRRuntimeFactory,
    ASRSessionSettings,
)
from application.audio_runtime import AudioRuntimeFactory, AudioRuntimePort
from application.audio_sources import AudioSourceFactory
from application.events import TranscriptSpeakerUpdateEvent
from application.recording import WavRecorder, WavRecorderFactory
from application.session_tasks import OfflinePassUseCase, StopAsrSessionUseCase
from asr.application.event_queue import AsrEventQueue
from audio.domain.formats import AudioFormat
from audio.domain.ports import AudioSource
from interface.session_controller_parts import (
    runtime_asr_mixin as _runtime_asr_ops,
)
from interface.session_controller_parts import (
    runtime_download_mixin as _runtime_download_ops,
)
from interface.session_controller_parts import (
    runtime_offline_pass_mixin as _runtime_offline_pass_ops,
)
from interface.session_controller_parts import (
    runtime_session_mixin as _runtime_session_ops,
)
from interface.session_controller_parts import source_mixin as _source_ops
from interface.session_controller_parts import (
    transcript_mixin as _transcript_ops,
)
from interface.session_controller_parts.settings import (
    asr_settings_from_params as _asr_settings_from_params,
)
from interface.session_controller_parts.types import EventSink, HeadlessSource
from session.domain.aggregate import SessionAggregate
from transcription.application.startup_service import (
    TranscriptionStartupService,
)
from transcription.application.transcript_store import TranscriptStore
from transcription.domain.aggregate import TranscriptionJobAggregate


class HeadlessSessionController:
    _write_session_outputs_locked = (
        _runtime_asr_ops._write_session_outputs_locked
    )
    _start_asr_locked = _runtime_asr_ops._start_asr_locked
    _resolve_stop_wav_path = _runtime_asr_ops._resolve_stop_wav_path
    _build_stop_asr_request = _runtime_asr_ops._build_stop_asr_request
    _stop_asr_locked = _runtime_asr_ops._stop_asr_locked
    _start_asr_event_pump = _runtime_asr_ops._start_asr_event_pump
    _run_asr_event_pump = _runtime_asr_ops._run_asr_event_pump
    _drain_asr_events = _runtime_asr_ops._drain_asr_events
    _drain_tap_queue = _runtime_asr_ops._drain_tap_queue
    begin_model_download = _runtime_download_ops.begin_model_download
    update_model_download_progress = (
        _runtime_download_ops.update_model_download_progress
    )
    finish_model_download = _runtime_download_ops.finish_model_download
    _should_run_offline_pass = (
        _runtime_offline_pass_ops._should_run_offline_pass
    )
    _start_offline_pass = _runtime_offline_pass_ops._start_offline_pass
    _run_offline_pass = _runtime_offline_pass_ops._run_offline_pass
    start_session = _runtime_session_ops.start_session
    stop_session = _runtime_session_ops.stop_session
    snapshot = _runtime_session_ops.snapshot
    transcript_text = _runtime_session_ops.transcript_text
    shutdown = _runtime_session_ops.shutdown
    _prepare_start_locked = _runtime_session_ops._prepare_start_locked
    _start_runtime_features_locked = (
        _runtime_session_ops._start_runtime_features_locked
    )
    _finalize_start_locked = _runtime_session_ops._finalize_start_locked
    _prepare_stop_params_locked = (
        _runtime_session_ops._prepare_stop_params_locked
    )
    _stop_runtime_locked = _runtime_session_ops._stop_runtime_locked
    _finalize_stop_locked = _runtime_session_ops._finalize_stop_locked
    _start_wav_if_requested = _runtime_session_ops._start_wav_if_requested
    _set_engine_output_enabled = _runtime_session_ops._set_engine_output_enabled
    _current_output_path = _runtime_session_ops._current_output_path
    _apply_tap_config = _runtime_session_ops._apply_tap_config
    set_event_sink = _source_ops.set_event_sink
    add_source = _source_ops.add_source
    remove_source = _source_ops.remove_source
    set_source_enabled = _source_ops.set_source_enabled
    set_source_delay = _source_ops.set_source_delay
    clear_transcript = _source_ops.clear_transcript
    _safe_meters = _source_ops._safe_meters
    _source_record = _source_ops._source_record
    _source_speaker_labels_locked = _source_ops._source_speaker_labels_locked
    _speaker_label_for_stream_locked = (
        _source_ops._speaker_label_for_stream_locked
    )
    _require_source = _source_ops._require_source
    _make_unique_name = _source_ops._make_unique_name
    _ensure_not_running = _source_ops._ensure_not_running
    _on_source_error = _source_ops._on_source_error
    _emit = _source_ops._emit
    _configure_transcript_files_locked = (
        _transcript_ops._configure_transcript_files_locked
    )
    _close_transcript_files_locked = (
        _transcript_ops._close_transcript_files_locked
    )
    _write_transcript_line_locked = (
        _transcript_ops._write_transcript_line_locked
    )
    _sync_transcript_store_paths_locked = (
        _transcript_ops._sync_transcript_store_paths_locked
    )
    _handle_asr_event = _transcript_ops._handle_asr_event
    handle_asr_event = _transcript_ops._handle_asr_event
    _dispatch_asr_event = _transcript_ops._dispatch_asr_event
    _handle_utterance_locked = _transcript_ops._handle_utterance_locked
    _handle_speaker_update_locked = (
        _transcript_ops._handle_speaker_update_locked
    )
    _handle_metrics_locked = _transcript_ops._handle_metrics_locked
    _handle_streaming_words_locked = (
        _transcript_ops._handle_streaming_words_locked
    )
    _handle_streaming_final_locked = (
        _transcript_ops._handle_streaming_final_locked
    )
    _append_transcript_line_locked = (
        _transcript_ops._append_transcript_line_locked
    )
    _emit_asr_notifications = _transcript_ops._emit_asr_notifications
    _update_line_speaker_from_event_locked = (
        _transcript_ops._update_line_speaker_from_event_locked
    )
    _find_line_for_speaker_update_locked = (
        _transcript_ops._find_line_for_speaker_update_locked
    )
    _remember_pending_speaker_update_locked = (
        _transcript_ops._remember_pending_speaker_update_locked
    )
    _apply_pending_speaker_updates_locked = (
        _transcript_ops._apply_pending_speaker_updates_locked
    )

    def __init__(
        self,
        *,
        project_root: Path,
        audio_runtime_factory: AudioRuntimeFactory,
        audio_source_factory: AudioSourceFactory,
        wav_recorder_factory: WavRecorderFactory,
        asr_runtime_factory: ASRRuntimeFactory | None = None,
        transcription_startup_service: TranscriptionStartupService
        | None = None,
        stop_asr_use_case: StopAsrSessionUseCase | None = None,
        offline_pass_use_case: OfflinePassUseCase | None = None,
        transcript_store: TranscriptStore | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.audio_runtime_factory = audio_runtime_factory
        self.audio_source_factory = audio_source_factory
        self.wav_recorder_factory = wav_recorder_factory
        self.asr_runtime_factory = asr_runtime_factory
        self.transcription_startup_service = transcription_startup_service
        self.stop_asr_use_case = (
            stop_asr_use_case
            if stop_asr_use_case is not None
            else StopAsrSessionUseCase()
        )
        self.offline_pass_use_case = offline_pass_use_case
        self.transcript_store = transcript_store
        self.event_sink = event_sink

        self.format = AudioFormat(
            sample_rate=48000, channels=2, dtype='float32', blocksize=1024
        )
        self.output_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=400)
        self.tap_queue: queue.Queue[dict] = queue.Queue(maxsize=200)
        self.asr_event_queue: queue.Queue[object] = AsrEventQueue(maxsize=600)
        self.engine: AudioRuntimePort = self.audio_runtime_factory.create(
            format=self.format,
            output_queue=self.output_queue,
            tap_queue=None,
        )
        self.writer: WavRecorder = self.wav_recorder_factory.create(
            self.output_queue
        )

        self._lock = threading.RLock()
        self._session_state = SessionAggregate()
        self._transcription_state = TranscriptionJobAggregate()
        self._sources: dict[str, HeadlessSource] = {}
        self._source_objs: dict[str, AudioSource] = {}
        self._status = 'idle'
        self._last_error = ''
        self._last_warning = ''
        self._asr_requested = False
        self._asr_running = False
        self._asr_runtime: ASRRuntime | None = None
        self._asr_event_stop = threading.Event()
        self._asr_event_thread: threading.Thread | None = None
        self._transcript_lines: list[dict[str, Any]] = []
        self._session_transcript_lines: list[dict[str, Any]] = []
        self._pending_speaker_updates: list[TranscriptSpeakerUpdateEvent] = []
        self._realtime_transcript_enabled = False
        self._human_log_path: Path | None = None
        self._realtime_transcript_path: Path | None = None
        self._asr_metrics: dict[str, Any] = {
            'segDroppedTotal': 0,
            'segSkippedTotal': 0,
            'avgLatencyS': 0.0,
            'p95LatencyS': 0.0,
            'lagS': 0.0,
            'queueWaitS': 0.0,
            'rtf': 0.0,
            'queueSize': 0,
            'device': '',
        }
        self._offline_pass_thread: threading.Thread | None = None
        self._offline_pass_running = False
        self._offline_pass_result: dict[str, Any] = {}
        self._model_download_info: dict[str, Any] = {}
        self._active_asr_settings: ASRSessionSettings | None = None
        self._active_asr_session_id: str = ''
        self._closing = False
        self._shutdown_complete = False
        self._writer_started = False

    def start(self) -> None:
        with self._lock:
            if self._writer_started:
                return
            self.writer.start()
            self._writer_started = True


__all__ = [
    'HeadlessSessionController',
    'HeadlessSource',
    '_asr_settings_from_params',
]
