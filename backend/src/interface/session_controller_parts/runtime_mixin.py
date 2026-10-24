from __future__ import annotations

from interface.session_controller_parts.runtime_asr_mixin import RuntimeAsrMixin
from interface.session_controller_parts.runtime_download_mixin import (
    RuntimeDownloadMixin,
)
from interface.session_controller_parts.runtime_offline_pass_mixin import (
    RuntimeOfflinePassMixin,
)
from interface.session_controller_parts.runtime_session_mixin import (
    RuntimeSessionMixin,
)


class RuntimeLifecycleMixin:
    """Compatibility facade with explicit operation bindings."""

    _write_session_outputs_locked = (
        RuntimeAsrMixin._write_session_outputs_locked
    )
    _start_asr_locked = RuntimeAsrMixin._start_asr_locked
    _resolve_stop_wav_path = RuntimeAsrMixin._resolve_stop_wav_path
    _build_stop_asr_request = RuntimeAsrMixin._build_stop_asr_request
    _stop_asr_locked = RuntimeAsrMixin._stop_asr_locked
    _start_asr_event_pump = RuntimeAsrMixin._start_asr_event_pump
    _run_asr_event_pump = RuntimeAsrMixin._run_asr_event_pump
    _drain_asr_events = RuntimeAsrMixin._drain_asr_events
    _drain_tap_queue = RuntimeAsrMixin._drain_tap_queue
    begin_model_download = RuntimeDownloadMixin.begin_model_download
    update_model_download_progress = (
        RuntimeDownloadMixin.update_model_download_progress
    )
    finish_model_download = RuntimeDownloadMixin.finish_model_download
    _should_run_offline_pass = RuntimeOfflinePassMixin._should_run_offline_pass
    _start_offline_pass = RuntimeOfflinePassMixin._start_offline_pass
    _run_offline_pass = RuntimeOfflinePassMixin._run_offline_pass
    start_session = RuntimeSessionMixin.start_session
    stop_session = RuntimeSessionMixin.stop_session
    snapshot = RuntimeSessionMixin.snapshot
    transcript_text = RuntimeSessionMixin.transcript_text
    shutdown = RuntimeSessionMixin.shutdown
    _prepare_start_locked = RuntimeSessionMixin._prepare_start_locked
    _start_runtime_features_locked = (
        RuntimeSessionMixin._start_runtime_features_locked
    )
    _finalize_start_locked = RuntimeSessionMixin._finalize_start_locked
    _prepare_stop_params_locked = (
        RuntimeSessionMixin._prepare_stop_params_locked
    )
    _stop_runtime_locked = RuntimeSessionMixin._stop_runtime_locked
    _finalize_stop_locked = RuntimeSessionMixin._finalize_stop_locked
    _start_wav_if_requested = RuntimeSessionMixin._start_wav_if_requested
    _set_engine_output_enabled = RuntimeSessionMixin._set_engine_output_enabled
    _current_output_path = RuntimeSessionMixin._current_output_path
    _apply_tap_config = RuntimeSessionMixin._apply_tap_config
