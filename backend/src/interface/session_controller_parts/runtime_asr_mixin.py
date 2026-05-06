from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Dict

from application.asr_language import runtime_asr_language
from application.session_tasks import StopAsrRequest
from interface.session_controller_parts.artifacts import generated_session_id, write_session_outputs
from interface.session_controller_parts.settings import asr_settings_from_params


class RuntimeAsrMixin:
    def _write_session_outputs_locked(
        self,
        *,
        asr: Any,
        settings,
        session_id: str,
    ) -> None:
        write_session_outputs(
            asr=asr,
            settings=settings,
            session_id=session_id,
            project_root=self.project_root,
            transcript_lines=self._transcript_lines,
        )

    def _start_asr_locked(self, params: Dict[str, Any]) -> str:
        if self.asr_runtime_factory is None or self.transcription_startup_service is None:
            self._last_warning = "Live ASR is not configured for the Electron backend yet."
            return self._last_warning

        self._drain_asr_events()
        settings = asr_settings_from_params(params, source_speaker_labels=self._source_speaker_labels_locked())
        self._active_asr_settings = settings
        self._transcription_state.begin_start(
            model_name=settings.model_name,
            mode=str(settings.mode),
            language=settings.language,
        )
        result = self.transcription_startup_service.start(
            settings,
            runtime_factory=self.asr_runtime_factory,
            tap_queue=self.tap_queue,
            project_root=self.project_root,
            event_queue=self.asr_event_queue,
        )

        for error in result.errors:
            self._emit("asr_start_attempt_failed", {"message": error})

        if not result.ok or result.asr is None or result.attempt is None:
            message = "; ".join(result.errors) or "unknown ASR startup error"
            self._transcription_state.fail_start(message)
            self._asr_running = False
            self._asr_runtime = None
            self._active_asr_session_id = ""
            self._last_warning = f"Live ASR could not start: {message}"
            return self._last_warning

        self._asr_runtime = result.asr
        self._asr_runtime.start()
        self._asr_running = True
        self._active_asr_session_id = str(getattr(result.asr, "session_id", "") or generated_session_id())
        self._transcription_state.finish_start(degraded=result.attempt.degraded, attempt_label=result.attempt.label)
        self._start_asr_event_pump()
        if result.degraded:
            self._last_warning = f"Live ASR started in fallback mode: {result.attempt.label}."
            return self._last_warning
        self._last_warning = ""
        return ""

    def _resolve_stop_wav_path(self, params: Dict[str, Any]) -> Path:
        return Path(str(
            params.get("wavPath")
            or self.writer.target_path()
            or self._current_output_path(str(params.get("outputFile", "")))
        ))

    def _build_stop_asr_request(self, asr: Any, wav_path: Path, params: Dict[str, Any]) -> StopAsrRequest:
        p = params or {}
        return StopAsrRequest(
            asr=asr,
            wav_path=wav_path,
            run_offline_pass=bool(p.get("runOfflinePass", False)),
            offline_model_name=str(p.get("offlineModelName", p.get("model", "large-v3"))),
            offline_language=runtime_asr_language(str(p.get("language", "ru"))),
        )

    def _stop_asr_locked(self, params: Dict[str, Any]) -> None:
        asr = self._asr_runtime
        self._asr_runtime = None
        self._asr_running = False
        self._asr_event_stop.set()
        self._transcript_lines = [l for l in self._transcript_lines if not str(l.get("id", "")).startswith("streaming-")]
        if asr is None:
            self._transcription_state.reset()
            self._active_asr_settings = None
            self._active_asr_session_id = ""
            return

        if self._transcription_state.can_stop:
            self._transcription_state.begin_stop()
        wav_path = self._resolve_stop_wav_path(params)
        result = self.stop_asr_use_case.execute(self._build_stop_asr_request(asr, wav_path, params))
        self._drain_asr_events()
        self._write_session_outputs_locked(
            asr=asr,
            settings=self._active_asr_settings,
            session_id=self._active_asr_session_id,
        )
        stop_error = str(result.stop_error or "")
        if self._transcription_state.is_stopping:
            self._transcription_state.finish_stop(stop_error)
        else:
            self._transcription_state.reset()
        self._active_asr_settings = None
        self._active_asr_session_id = ""
        if stop_error:
            self._last_error = stop_error
            self._emit("asr_stop_error", {"message": stop_error})

    def _start_asr_event_pump(self) -> None:
        self._asr_event_stop = threading.Event()
        self._asr_event_thread = threading.Thread(target=self._run_asr_event_pump, name="electron-asr-events", daemon=True)
        self._asr_event_thread.start()

    def _run_asr_event_pump(self) -> None:
        while not self._asr_event_stop.is_set():
            try:
                item = self.asr_event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle_asr_event(item)

    def _drain_asr_events(self, *, limit: int = 500) -> int:
        count = 0
        while count < limit:
            try:
                item = self.asr_event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_asr_event(item)
            count += 1
        return count

    def _drain_tap_queue(self) -> int:
        count = 0
        while True:
            try:
                self.tap_queue.get_nowait()
                count += 1
            except queue.Empty:
                return count
