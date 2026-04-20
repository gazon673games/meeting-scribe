from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any, List, Optional

from PySide6.QtCore import QTimer

from application.asr_language import initial_prompt_for_language, normalize_asr_language, runtime_asr_language
from application.asr_session import ASRSessionSettings
from application.command_bus import CommandDispatcher
from application.commands import StartSessionCommand, StopSessionCommand, SwitchProfileCommand
from application.event_types import (
    AsrStopDoneEvent,
    OfflinePassDoneEvent,
    OfflinePassErrorEvent,
    OfflinePassStartedEvent,
    event_from_record,
)
from application.model_download import download_model_async, is_model_cached
from application.session_tasks import OfflinePassRequest, StopAsrRequest
from session.domain.aggregate import SessionAggregate
from transcription.domain.aggregate import TranscriptionJobAggregate


class SessionMixin:
    def _init_session_state(self) -> None:
        self.asr: Any = None
        self.asr_running: bool = False
        self._asr_overload_active: bool = False
        self._last_warn_ts: float = 0.0

        # session metrics mirror (UI side)
        self._tap_dropped_total: int = 0
        self._seg_dropped_total: int = 0
        self._seg_skipped_total: int = 0
        self._avg_latency_s: float = 0.0
        self._p95_latency_s: float = 0.0
        self._lag_s: float = 0.0

        # silence alert tracking
        self._silence_eps: float = 1e-4
        self._silence_alert_s: float = 15.0
        self._desktop_silence_since_mono: Optional[float] = None

        # UI timing modes
        self._long_run_mode: bool = False
        self._ui_interval_normal_ms: int = 120
        self._ui_interval_long_ms: int = 260

        # lifecycle state
        self._session_state = SessionAggregate()
        self._transcription_state = TranscriptionJobAggregate()
        self._asr_supervision_report = None
        self._closing: bool = False
        self._offline_thread: Any = None
        self._asr_stop_thread: Any = None
        self._command_dispatcher = CommandDispatcher()
        self._command_dispatcher.register(StartSessionCommand, self._handle_start_session_command)
        self._command_dispatcher.register(StopSessionCommand, self._handle_stop_session_command)
        self._command_dispatcher.register(SwitchProfileCommand, self._handle_switch_profile_command)

    def _start_all(self) -> None:
        command = StartSessionCommand(
            source_count=len(self.rows),
            asr_enabled=bool(self.chk_asr.isChecked()),
            model_name=self.cmb_model.currentText().strip() or "medium",
            profile=str(self.cmb_profile.currentText()),
            language=normalize_asr_language(self.cmb_lang.currentText()),
        )
        self._command_dispatcher.dispatch(command)

    def _handle_start_session_command(self, command: StartSessionCommand) -> None:
        if self._is_running():
            return
        if not self._session_state.can_start:
            if self._session_state.is_stopping:
                self._set_status("ASR is still stopping. Wait for it to finish first.")
                return
            if self._session_state.is_offline_pass:
                self._set_status("Offline pass is still running. Wait for it to finish first.")
                return
            if self._session_state.is_downloading_model:
                self._set_status("Model download is still running. Wait for it to finish first.")
                return
            self._set_status(f"Session is busy: {self._session_state.state.value}")
            return
        if command.source_count == 0:
            self._set_status("Add at least one device first.")
            return

        if command.asr_enabled:
            if not is_model_cached(command.model_name):
                self._download_model_then_start(command.model_name)
                return

        self._session_state.begin_start(
            source_count=command.source_count,
            asr_enabled=command.asr_enabled,
            profile=command.profile,
            language=command.language,
        )

        self._human_log_close()
        self._reset_session_metrics()
        self._desktop_silence_since_mono = None

        while True:
            try:
                self.asr_ui_q.get_nowait()
            except queue.Empty:
                break

        for name in list(self.rows.keys()):
            self._apply_delay_from_ui(name)

        enabled_sources: List[str] = [name for name, row in self.rows.items() if row.enabled.isChecked()]

        self._rt_tr_to_file = bool(self.chk_rt_transcript_file.isChecked())
        if not self._rt_tr_to_file:
            self._rt_close()

        if command.asr_enabled:
            self.engine.set_tap_queue(self.tap_q)

            mode = "split" if self.cmb_asr_mode.currentIndex() == 1 else "mix"
            if mode == "mix":
                self.engine.set_tap_config(mode="mix", sources=None, drop_threshold=0.85)
            else:
                self.engine.set_tap_config(mode="sources", sources=enabled_sources, drop_threshold=0.85)
        else:
            self.engine.set_tap_queue(None)

        try:
            self.engine.start()
        except Exception as e:
            self._session_state.fail_start(str(e))
            self._set_status(f"Engine start failed: {e}")
            return

        QTimer.singleShot(1400, self._post_start_audio_check)

        self.asr_running = False
        self.asr = None
        self._start_asr_if_enabled()
        self._start_wav_if_enabled()
        self._set_session_controls_running()

        self._session_state.finish_start(
            asr_running=bool(self.asr_running),
            wav_recording=bool(self.writer.is_recording()),
        )
        self.ui_timer.start()
        self._set_status(f"running: ASR={'on' if self.asr_running else 'off'}, WAV={'on' if self.writer.is_recording() else 'off'}")

    def _download_model_then_start(self, model_name: str) -> None:
        self._session_state.begin_model_download(model_name)
        self.btn_start.setEnabled(False)
        self._set_status(f"Downloading model {model_name}... please wait")

        def on_progress(msg: str) -> None:
            self._set_status(msg)

        def on_done(error: Optional[str]) -> None:
            self._session_state.finish_model_download(model_name, error=str(error or ""))
            if error:
                self._set_status(f"Model download failed: {error}")
                self.btn_start.setEnabled(True)
            else:
                self._set_status(f"Model {model_name} ready.")
                self._start_all()

        download_model_async(model_name, on_progress=on_progress, on_done=on_done)

    def _reset_session_metrics(self) -> None:
        self._asr_overload_active = False
        self._last_warn_ts = 0.0
        self._tap_dropped_total = 0
        self._seg_dropped_total = 0
        self._seg_skipped_total = 0
        self._avg_latency_s = 0.0
        self._p95_latency_s = 0.0
        self._lag_s = 0.0

    def _start_asr_if_enabled(self) -> None:
        if not self.chk_asr.isChecked():
            return

        settings = self._read_asr_session_settings()
        self._transcription_state.begin_start(
            model_name=settings.model_name,
            mode=str(settings.mode),
            language=settings.language,
        )
        result = self.transcription_startup_service.start(
            settings,
            runtime_factory=self.asr_runtime_factory,
            tap_queue=self.tap_q,
            project_root=self.project_root,
            event_queue=self.asr_ui_q,
        )
        self._asr_supervision_report = result.supervision_report

        for msg in result.errors:
            self._append_transcript_line(f"[{self._fmt_ts(time.time())}] ASR start failed: {msg}")

        if result.ok and result.attempt is not None:
            attempt = result.attempt
            if result.errors:
                self._transcription_state.begin_fallback(
                    attempt_label=attempt.label,
                    model_name=attempt.settings.model_name,
                    reason=result.errors[-1],
                )
                self._append_transcript_line(
                    f"[{self._fmt_ts(time.time())}] ASR fallback selected: {attempt.label} "
                    f"({attempt.settings.model_name}, {attempt.settings.device}, {attempt.settings.compute_type})"
                )
            self.asr = result.asr
            self.asr_running = True
            self._transcription_state.finish_start(degraded=attempt.degraded, attempt_label=attempt.label)

            human_log_path = self._human_log_open_session()
            if human_log_path is not None:
                self._append_transcript_line(f"[{self._fmt_ts(time.time())}] human log -> {human_log_path}")
            if attempt.degraded:
                self._append_transcript_line(
                    f"[{self._fmt_ts(time.time())}] ASR running in degraded mode via {attempt.label}"
                )
            return

        self._transcription_state.fail_start("; ".join(result.errors))
        self.asr = None
        self.asr_running = False
        self._set_status("ASR start failed after fallback attempts. Audio session continues without ASR.")

    def _read_asr_session_settings(self) -> ASRSessionSettings:
        mode = "split" if self.cmb_asr_mode.currentIndex() == 1 else "mix"
        model_name = self.cmb_model.currentText().strip() or "medium"

        lang_ui = normalize_asr_language(self.cmb_lang.currentText())
        asr_lang = runtime_asr_language(lang_ui)
        prompt = initial_prompt_for_language(lang_ui)

        compute_type = str(self.cmb_compute.currentText() or "float16")
        beam_size = self._safe_int(self.txt_beam.text(), 5, 1, 20)
        endpoint_silence_ms = self._safe_float(self.txt_endpoint.text(), 650.0, 50.0, 5000.0)
        max_segment_s = self._safe_float(self.txt_maxseg.text(), 7.0, 1.0, 60.0)
        overlap_ms = self._safe_float(self.txt_overlap.text(), 200.0, 0.0, 2000.0)
        vad_thr = self._safe_float(self.txt_vad_thr.text(), 0.0055, 1e-5, 1.0)

        overload_strategy = str(self.cmb_overload_strategy.currentText() or "drop_old").strip().lower()
        overload_enter = self._safe_int(self.txt_over_enter.text(), 18, 1, 999)
        overload_exit = self._safe_int(self.txt_over_exit.text(), 6, 1, 999)
        overload_hard = self._safe_int(self.txt_over_hard.text(), 28, 1, 999)
        overload_beamcap = self._safe_int(self.txt_over_beamcap.text(), 2, 1, 20)
        overload_maxseg = self._safe_float(self.txt_over_maxseg.text(), 5.0, 0.5, 60.0)
        overload_overlap = self._safe_float(self.txt_over_overlap.text(), 120.0, 0.0, 2000.0)

        return ASRSessionSettings(
            language=lang_ui,
            mode=mode,
            model_name=model_name,
            device="cuda",
            compute_type=compute_type,
            beam_size=beam_size,
            endpoint_silence_ms=endpoint_silence_ms,
            max_segment_s=max_segment_s,
            overlap_ms=overlap_ms,
            vad_energy_threshold=vad_thr,
            overload_strategy="keep_all" if overload_strategy == "keep_all" else "drop_old",
            overload_enter_qsize=overload_enter,
            overload_exit_qsize=overload_exit,
            overload_hard_qsize=overload_hard,
            overload_beam_cap=overload_beamcap,
            overload_max_segment_s=overload_maxseg,
            overload_overlap_ms=overload_overlap,
            asr_language=asr_lang,
            asr_initial_prompt=prompt,
        )

    def _start_wav_if_enabled(self) -> None:
        if not self.chk_wav.isChecked():
            return
        if not self._wav_recording_available():
            self._set_status("WAV disabled: install soundfile.")
            return

        out_path = self._current_output_path()
        self.output_name = out_path.name
        try:
            self.writer.start_recording(out_path, self.fmt)
        except Exception as e:
            self._set_status(f"WAV start failed: {e}")

    def _set_session_controls_running(self) -> None:
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.btn_add.setEnabled(False)
        self.chk_asr.setEnabled(False)
        self.cmb_profile.setEnabled(False)
        self.cmb_lang.setEnabled(False)
        self.cmb_asr_mode.setEnabled(False)
        self.cmb_model.setEnabled(False)
        self.grp_asr_cfg.setEnabled(False)
        self.btn_asr_toggle.setEnabled(False)
        self.chk_longrun.setEnabled(False)

        self.chk_offline_on_stop.setEnabled(False)
        self.chk_rt_transcript_file.setEnabled(False)

        self.chk_wav.setEnabled(False)
        self.txt_output.setEnabled(False)

    def _post_start_audio_check(self) -> None:
        if not self._is_running():
            return

        meters = self.engine.get_meters()
        srcs = meters.get("sources", {})
        now_mono = time.monotonic()

        failed: List[str] = []
        silent: List[str] = []
        for name, row in self.rows.items():
            if not row.enabled.isChecked():
                continue

            info = srcs.get(name, {})
            last_ts = float(info.get("last_ts", 0.0))
            rms = float(info.get("rms", 0.0))
            if last_ts <= 0.0 or ((now_mono - last_ts) > 2.0 and rms <= 1e-5):
                silent.append(name)

            src_obj = self.source_objs.get(name)
            if src_obj is not None and hasattr(src_obj, "get_last_error"):
                try:
                    err = src_obj.get_last_error()
                except Exception:
                    err = None
                if err:
                    failed.append(f"{name}: {err}")

        if failed:
            for msg in failed:
                self._append_transcript_line(f"[{self._fmt_ts(time.time())}] SOURCE ERROR: {msg}")
            self._set_status("Audio source failed. See transcript for details.")
            return

        if silent:
            joined = ", ".join(silent[:3])
            more = "..." if len(silent) > 3 else ""
            self._warn_throttle(
                f"No audio frames detected from: {joined}{more}. "
                "Try re-adding loopback device and Start again.",
                min_interval_s=0.1,
            )

    def _set_stop_ui_pending(self) -> None:
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)

        self.btn_add.setEnabled(False)
        self.chk_asr.setEnabled(False)
        self.cmb_profile.setEnabled(False)
        self.cmb_lang.setEnabled(False)
        self.cmb_asr_mode.setEnabled(False)
        self.cmb_model.setEnabled(False)
        self.grp_asr_cfg.setEnabled(False)
        self.btn_asr_toggle.setEnabled(False)
        self.chk_longrun.setEnabled(False)
        self.chk_offline_on_stop.setEnabled(False)
        self.chk_rt_transcript_file.setEnabled(False)
        self.chk_wav.setEnabled(False)
        self.txt_output.setEnabled(False)

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

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        self.btn_add.setEnabled(True)
        self.chk_asr.setEnabled(True)
        self.cmb_profile.setEnabled(True)
        self.cmb_lang.setEnabled(True)
        self.cmb_asr_mode.setEnabled(True)
        self.cmb_model.setEnabled(True)
        self.grp_asr_cfg.setEnabled(True)
        self.btn_asr_toggle.setEnabled(True)
        self.chk_longrun.setEnabled(True)

        self.chk_offline_on_stop.setEnabled(self._offline_asr_available())
        self.chk_rt_transcript_file.setEnabled(True)

        self._set_custom_enabled(self.cmb_profile.currentText() == self.PROFILE_CUSTOM)

        self.chk_wav.setEnabled(self._wav_recording_available())
        self.txt_output.setEnabled(True)

        self._set_progress_if_changed(self.master_meter, 0)
        self._set_label_text_if_changed(self.master_status, "stopped")
        self._set_label_text_if_changed(self.lbl_drops, "drops: 0")
        for row in self.rows.values():
            self._set_progress_if_changed(row.meter, 0)
            self._set_label_text_if_changed(row.status, "stopped")

        self._drain_asr_ui_events(limit=500)

        self._rt_close()
        self._human_log_close()

        if stop_error:
            self._append_transcript_line(f"[{self._fmt_ts(time.time())}] ASR stop error: {stop_error}")

        wav_error = self.writer.last_error()
        if stop_error:
            self._set_status(f"stopped (asr stop error: {stop_error})")
        elif wav_error:
            self._set_status(f"stopped (wav error: {wav_error})")
        else:
            self._set_status("stopped")

        self._flush_config_if_dirty()

        if (
            not stop_error
            and run_offline_pass
            and self._offline_asr_available()
            and bool(self.chk_offline_on_stop.isChecked())
            and self._wav_recording_available()
            and Path(wav_path).exists()
        ):
            self._start_offline_pass(
                Path(wav_path),
                model_name=offline_model_name,
                language=offline_language,
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
        result = self.stop_asr_use_case.execute(
            StopAsrRequest(
                asr=asr_obj,
                wav_path=Path(wav_path),
                run_offline_pass=bool(run_offline_pass),
                offline_model_name=str(offline_model_name),
                offline_language=offline_language,
            )
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

    def _stop_all(self, *, run_offline_pass: bool = True, wait: bool = False) -> None:
        command = StopSessionCommand(run_offline_pass=bool(run_offline_pass), wait=bool(wait))
        self._command_dispatcher.dispatch(command)

    def _handle_stop_session_command(self, command: StopSessionCommand) -> None:
        if self._session_state.is_stopping:
            if command.wait and self._asr_stop_thread is not None:
                self._asr_stop_thread.join()
            else:
                self._set_status("ASR is still stopping. Wait for it to finish.")
            return
        if self._session_state.is_offline_pass:
            self._set_status("Offline pass is still running. Wait for it to finish.")
            return
        if not self._session_state.can_stop:
            if not self.engine.is_running() and self.asr is None and not self.writer.is_recording():
                return
            self._set_status(f"Cannot stop session while state is {self._session_state.state.value}.")
            return

        self._session_state.begin_stop(run_offline_pass=command.run_offline_pass)

        wav_path = self.writer.target_path() or self._current_output_path()
        offline_model_name = str(self.cmb_model.currentText() or "large-v3")
        lang_ui = normalize_asr_language(self.cmb_lang.currentText())
        offline_language = runtime_asr_language(lang_ui)

        if self.writer.is_recording():
            self.writer.stop_recording()

        asr_to_stop = self.asr
        self.asr = None
        self.asr_running = False
        if self._transcription_state.can_stop:
            self._transcription_state.begin_stop()

        if self.engine.is_running():
            try:
                self.engine.stop()
            except Exception as e:
                self._set_status(f"Engine stop error: {e}")

        try:
            self.engine.set_tap_queue(None)
        except Exception:
            pass

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
            result = self.stop_asr_use_case.execute(
                StopAsrRequest(
                    asr=asr_to_stop,
                    wav_path=Path(wav_path),
                    run_offline_pass=bool(command.run_offline_pass),
                    offline_model_name=str(offline_model_name),
                    offline_language=offline_language,
                )
            )
            self._finish_stop_ui(
                wav_path=result.wav_path,
                run_offline_pass=result.run_offline_pass,
                offline_model_name=result.offline_model_name,
                offline_language=result.offline_language,
                stop_error=result.stop_error,
            )
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

    def _start_offline_pass(self, wav_path: Path, *, model_name: str, language: Optional[str]) -> None:
        if self._session_state.is_offline_pass:
            self._set_status("Offline pass is already running.")
            return
        if not self._offline_asr_available():
            self._set_status("Offline pass is unavailable.")
            return
        if not self._session_state.can_start:
            self._set_status(f"Cannot start offline pass while state is {self._session_state.state.value}.")
            return

        self._session_state.begin_offline_pass(str(wav_path))
        self.btn_start.setEnabled(False)

        self._offline_thread = self.background_task_runner.start(
            name="offline-asr-pass",
            target=self._run_offline_pass_worker,
            args=(Path(wav_path), str(model_name), language),
        )

    def _run_offline_pass_worker(self, wav_path: Path, model_name: str, language: Optional[str]) -> None:
        try:
            self.background_event.emit(OfflinePassStartedEvent())

            result = self.offline_pass_use_case.execute(
                OfflinePassRequest(
                    project_root=self.project_root,
                    wav_path=Path(wav_path),
                    model_name=str(model_name or "large-v3"),
                    language=language,
                )
            )

            self.background_event.emit(OfflinePassDoneEvent(out_txt=str(result.out_txt)))
        except Exception as e:
            self.background_event.emit(OfflinePassErrorEvent(error=f"{type(e).__name__}: {e}"))

    def _handle_background_event(self, ev: object) -> None:
        if self._closing:
            return

        event = event_from_record(ev)
        if isinstance(event, AsrStopDoneEvent):
            stop_error_raw = event.stop_error
            self._finish_stop_ui(
                wav_path=Path(str(event.wav_path or self._current_output_path())),
                run_offline_pass=bool(event.run_offline_pass),
                offline_model_name=str(event.offline_model_name or self.cmb_model.currentText() or "large-v3"),
                offline_language=event.offline_language,
                stop_error=(str(stop_error_raw).strip() if stop_error_raw is not None else None),
            )
            return

        if isinstance(event, OfflinePassStartedEvent):
            self._append_transcript_line(f"[{self._fmt_ts(time.time())}] offline pass: starting...")
            self._set_status("offline pass: running")
            return

        if isinstance(event, OfflinePassDoneEvent):
            if self._session_state.is_offline_pass:
                self._session_state.finish_offline_pass(out_txt=str(event.out_txt))
            self._offline_thread = None
            self.btn_start.setEnabled(True)
            out_txt = str(event.out_txt).strip()
            self._append_transcript_line(f"[{self._fmt_ts(time.time())}] offline pass: done -> {out_txt}")
            self._set_status("offline pass: done")
            return

        if isinstance(event, OfflinePassErrorEvent):
            if self._session_state.is_offline_pass:
                self._session_state.finish_offline_pass(error=str(event.error or "unknown error"))
            self._offline_thread = None
            self.btn_start.setEnabled(True)
            err = str(event.error or "unknown error")
            self._append_transcript_line(f"[{self._fmt_ts(time.time())}] offline pass: ERROR {err}")
            self._set_status("offline pass: failed")
            return
