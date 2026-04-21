from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from application.asr_language import normalize_asr_language, runtime_asr_language
from application.commands import StopSessionCommand
from application.event_types import (
    AsrStopDoneEvent,
    OfflinePassDoneEvent,
    OfflinePassErrorEvent,
    OfflinePassStartedEvent,
    event_from_record,
)
from application.session_tasks import StopAsrRequest
from ui.session_offline_mixin import SessionOfflineMixin
from ui.session_ui_mixin import SessionUiMode


class SessionStopMixin(SessionOfflineMixin):
    def _finish_stop_ui(
        self,
        *,
        wav_path: Path,
        run_offline_pass: bool,
        offline_model_name: str,
        offline_language: Optional[str],
        stop_error: Optional[str] = None,
    ) -> None:
        if self._session_state.is_stopping:
            self._session_state.finish_stop(stop_error or "")
        if self._transcription_state.is_stopping:
            self._transcription_state.finish_stop(stop_error or "")
        elif not self.asr_running:
            self._transcription_state.reset()

        self._asr_stop_thread = None
        self.ui_timer.stop()
        self._apply_session_ui_mode(SessionUiMode.IDLE)
        self._reset_stop_meters()
        self._drain_asr_ui_events(limit=500)
        self._rt_close()
        self._human_log_close()

        if stop_error:
            self._append_transcript_line(f"[{self._fmt_ts(time.time())}] ASR stop error: {stop_error}")
        self._set_stop_status(stop_error=stop_error)
        self._flush_config_if_dirty()

        if self._should_run_offline_pass(stop_error=stop_error, run_offline_pass=run_offline_pass, wav_path=wav_path):
            self._start_offline_pass(Path(wav_path), model_name=offline_model_name, language=offline_language)

    def _reset_stop_meters(self) -> None:
        self._set_progress_if_changed(self.master_meter, 0)
        self._set_label_text_if_changed(self.master_status, "stopped")
        self._set_label_text_if_changed(self.lbl_drops, "drops: 0")
        for row in self.rows.values():
            self._set_progress_if_changed(row.meter, 0)
            self._set_label_text_if_changed(row.status, "stopped")

    def _set_stop_status(self, *, stop_error: Optional[str]) -> None:
        wav_error = self.writer.last_error()
        if stop_error:
            self._set_status(f"stopped (asr stop error: {stop_error})")
        elif wav_error:
            self._set_status(f"stopped (wav error: {wav_error})")
        else:
            self._set_status("stopped")

    def _should_run_offline_pass(self, *, stop_error: Optional[str], run_offline_pass: bool, wav_path: Path) -> bool:
        return bool(
            not stop_error
            and run_offline_pass
            and self._offline_asr_available()
            and self.chk_offline_on_stop.isChecked()
            and self._wav_recording_available()
            and Path(wav_path).exists()
        )

    def _stop_all(self, *, run_offline_pass: bool = True, wait: bool = False) -> None:
        self._command_dispatcher.dispatch(StopSessionCommand(run_offline_pass=bool(run_offline_pass), wait=bool(wait)))

    def _handle_stop_session_command(self, command: StopSessionCommand) -> None:
        if self._stop_is_blocked(command):
            return

        self._session_state.begin_stop(run_offline_pass=command.run_offline_pass)
        wav_path, offline_model_name, offline_language = self._read_stop_plan()

        if self.writer.is_recording():
            self.writer.stop_recording()

        asr_to_stop = self._detach_asr_for_stop()
        self._stop_engine_for_session()
        self.ui_timer.stop()
        self._set_stop_ui_pending()

        if asr_to_stop is None:
            self._finish_stop_ui(
                wav_path=Path(wav_path),
                run_offline_pass=command.run_offline_pass,
                offline_model_name=offline_model_name,
                offline_language=offline_language,
            )
            return

        self._set_status("stopping (waiting for ASR to finish current transcription)...")
        if command.wait or self._closing:
            self._stop_asr_synchronously(asr_to_stop, wav_path, command, offline_model_name, offline_language)
            return

        self._asr_stop_thread = self.background_task_runner.start(
            name="asr-stop-worker",
            target=self._run_asr_stop_worker,
            kwargs={
                "asr_obj": asr_to_stop,
                "wav_path": Path(wav_path),
                "run_offline_pass": command.run_offline_pass,
                "offline_model_name": offline_model_name,
                "offline_language": offline_language,
            },
        )

    def _stop_is_blocked(self, command: StopSessionCommand) -> bool:
        if self._session_state.is_stopping:
            if command.wait and self._asr_stop_thread is not None:
                self._asr_stop_thread.join()
            else:
                self._set_status("ASR is still stopping. Wait for it to finish.")
            return True
        if self._session_state.is_offline_pass:
            self._set_status("Offline pass is still running. Wait for it to finish.")
            return True
        if not self._session_state.can_stop:
            already_idle = not self.engine.is_running() and self.asr is None and not self.writer.is_recording()
            if not already_idle:
                self._set_status(f"Cannot stop session while state is {self._session_state.state.value}.")
            return True
        return False

    def _execute_stop_request(
        self,
        asr_obj: Any,
        *,
        wav_path: Path,
        run_offline_pass: bool,
        offline_model_name: str,
        offline_language: Optional[str],
    ):
        return self.stop_asr_use_case.execute(
            StopAsrRequest(
                asr=asr_obj,
                wav_path=Path(wav_path),
                run_offline_pass=bool(run_offline_pass),
                offline_model_name=str(offline_model_name),
                offline_language=offline_language,
            )
        )

    def _run_asr_stop_worker(
        self,
        asr_obj: Any,
        *,
        wav_path: Path,
        run_offline_pass: bool,
        offline_model_name: str,
        offline_language: Optional[str],
    ) -> None:
        result = self._execute_stop_request(
            asr_obj,
            wav_path=wav_path,
            run_offline_pass=run_offline_pass,
            offline_model_name=offline_model_name,
            offline_language=offline_language,
        )
        self.background_event.emit(
            AsrStopDoneEvent(
                wav_path=str(result.wav_path),
                run_offline_pass=bool(result.run_offline_pass),
                offline_model_name=str(result.offline_model_name),
                offline_language=result.offline_language,
                stop_error=result.stop_error,
            )
        )

    def _stop_asr_synchronously(
        self,
        asr_to_stop: Any,
        wav_path: Path,
        command: StopSessionCommand,
        offline_model_name: str,
        offline_language: Optional[str],
    ) -> None:
        result = self._execute_stop_request(
            asr_to_stop,
            wav_path=wav_path,
            run_offline_pass=command.run_offline_pass,
            offline_model_name=offline_model_name,
            offline_language=offline_language,
        )
        self._finish_stop_ui(
            wav_path=result.wav_path,
            run_offline_pass=result.run_offline_pass,
            offline_model_name=result.offline_model_name,
            offline_language=result.offline_language,
            stop_error=result.stop_error,
        )

    def _read_stop_plan(self) -> tuple[Path, str, Optional[str]]:
        wav_path = self.writer.target_path() or self._current_output_path()
        offline_model_name = str(self.cmb_model.currentText() or "large-v3")
        lang_ui = normalize_asr_language(self.cmb_lang.currentText())
        return Path(wav_path), offline_model_name, runtime_asr_language(lang_ui)

    def _detach_asr_for_stop(self) -> Any:
        asr_to_stop = self.asr
        self.asr = None
        self.asr_running = False
        if self._transcription_state.can_stop:
            self._transcription_state.begin_stop()
        return asr_to_stop

    def _stop_engine_for_session(self) -> None:
        if self.engine.is_running():
            try:
                self.engine.stop()
            except Exception as e:
                self._set_status(f"Engine stop error: {e}")
        try:
            self.engine.set_tap_queue(None)
        except Exception:
            pass

    def _handle_background_event(self, ev: object) -> None:
        if self._closing:
            return
        event = event_from_record(ev)
        if isinstance(event, AsrStopDoneEvent):
            self._handle_asr_stop_done_event(event)
        elif isinstance(event, OfflinePassStartedEvent):
            self._handle_offline_pass_started()
        elif isinstance(event, OfflinePassDoneEvent):
            self._handle_offline_pass_done_event(event)
        elif isinstance(event, OfflinePassErrorEvent):
            self._handle_offline_pass_error_event(event)

    def _handle_asr_stop_done_event(self, event: AsrStopDoneEvent) -> None:
        stop_error = str(event.stop_error).strip() if event.stop_error is not None else None
        self._finish_stop_ui(
            wav_path=Path(str(event.wav_path or self._current_output_path())),
            run_offline_pass=bool(event.run_offline_pass),
            offline_model_name=str(event.offline_model_name or self.cmb_model.currentText() or "large-v3"),
            offline_language=event.offline_language,
            stop_error=stop_error,
        )
