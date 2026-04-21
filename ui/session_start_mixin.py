from __future__ import annotations

import queue
import time
from typing import List, Optional

from PySide6.QtCore import QTimer

from application.asr_language import initial_prompt_for_language, normalize_asr_language, runtime_asr_language
from application.asr_session import ASRSessionSettings
from application.commands import StartSessionCommand
from application.model_download import download_model_async, is_model_cached
from ui.asr_field_defs import _ASR_FLOAT_FIELDS, _ASR_INT_FIELDS


class SessionStartMixin:
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

        start_error = self._start_block_reason(command)
        if start_error:
            self._set_status(start_error)
            return

        if command.asr_enabled and not is_model_cached(command.model_name):
            self._download_model_then_start(command.model_name)
            return

        self._session_state.begin_start(
            source_count=command.source_count,
            asr_enabled=command.asr_enabled,
            profile=command.profile,
            language=command.language,
        )

        self._prepare_session_start()
        self._configure_tap_for_start(command)

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
        wav_error = self._start_wav_if_enabled()
        self._finish_start_ui(wav_error=wav_error)

    def _start_block_reason(self, command: StartSessionCommand) -> Optional[str]:
        if not self._session_state.can_start:
            if self._session_state.is_stopping:
                return "ASR is still stopping. Wait for it to finish first."
            if self._session_state.is_offline_pass:
                return "Offline pass is still running. Wait for it to finish first."
            if self._session_state.is_downloading_model:
                return "Model download is still running. Wait for it to finish first."
            return f"Session is busy: {self._session_state.state.value}"
        if command.source_count == 0:
            return "Add at least one device first."
        return None

    def _prepare_session_start(self) -> None:
        self._human_log_close()
        self._reset_session_metrics()
        self._desktop_silence_since_mono = None
        self._drain_asr_event_queue()
        for name in list(self.rows.keys()):
            self._apply_delay_from_ui(name)
        self._rt_tr_to_file = bool(self.chk_rt_transcript_file.isChecked())
        if not self._rt_tr_to_file:
            self._rt_close()

    def _drain_asr_event_queue(self) -> None:
        while True:
            try:
                self.asr_ui_q.get_nowait()
            except queue.Empty:
                break

    def _configure_tap_for_start(self, command: StartSessionCommand) -> None:
        if not command.asr_enabled:
            self.engine.set_tap_queue(None)
            return
        self.engine.set_tap_queue(self.tap_q)
        if self.cmb_asr_mode.currentIndex() == 1:
            self.engine.set_tap_config(mode="sources", sources=self._enabled_source_names(), drop_threshold=0.85)
        else:
            self.engine.set_tap_config(mode="mix", sources=None, drop_threshold=0.85)

    def _enabled_source_names(self) -> List[str]:
        return [name for name, row in self.rows.items() if row.enabled.isChecked()]

    def _finish_start_ui(self, *, wav_error: Optional[str] = None) -> None:
        self._set_session_controls_running()
        self._session_state.finish_start(
            asr_running=bool(self.asr_running),
            wav_recording=bool(self.writer.is_recording()),
        )
        self.ui_timer.start()
        status = f"running: ASR={'on' if self.asr_running else 'off'}, WAV={'on' if self.writer.is_recording() else 'off'}"
        if wav_error:
            status = f"{status}; {wav_error}"
        self._set_status(status)

    def _download_model_then_start(self, model_name: str) -> None:
        self._session_state.begin_model_download(model_name)
        self.btn_start.setEnabled(False)
        self._set_status(f"Downloading model {model_name}... please wait")
        download_model_async(
            model_name,
            on_progress=self._set_status,
            on_done=lambda error: self._on_model_download_done(model_name, error),
        )

    def _on_model_download_done(self, model_name: str, error: Optional[str]) -> None:
        self._session_state.finish_model_download(model_name, error=str(error or ""))
        if error:
            self._set_status(f"Model download failed: {error}")
            self.btn_start.setEnabled(True)
        else:
            self._set_status(f"Model {model_name} ready.")
            self._start_all()

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
            self._append_transcript_line(f"[{self._fmt_ts(time.time())}] ASR attempt failed: {msg}")

        if result.ok and result.attempt is not None:
            self._finish_asr_start_success(result)
            return

        self._transcription_state.fail_start("; ".join(result.errors))
        self.asr = None
        self.asr_running = False
        self._set_status("ASR start failed after fallback attempts. Audio session continues without ASR.")

    def _finish_asr_start_success(self, result) -> None:
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

    def _read_asr_session_settings(self) -> ASRSessionSettings:
        lang_ui = normalize_asr_language(self.cmb_lang.currentText())
        overload_strategy = str(self.cmb_overload_strategy.currentText() or "drop_old").strip().lower()
        kw: dict = {}
        for key, attr, default, lo, hi in _ASR_INT_FIELDS:
            kw[key] = self._safe_int(getattr(self, attr).text(), default, lo, hi)
        for key, attr, default, lo, hi in _ASR_FLOAT_FIELDS:
            kw[key] = self._safe_float(getattr(self, attr).text(), default, lo, hi)
        return ASRSessionSettings(
            language=lang_ui,
            mode="split" if self.cmb_asr_mode.currentIndex() == 1 else "mix",
            model_name=self.cmb_model.currentText().strip() or "medium",
            device="cuda",
            compute_type=str(self.cmb_compute.currentText() or "float16"),
            overload_strategy="keep_all" if overload_strategy == "keep_all" else "drop_old",
            asr_language=runtime_asr_language(lang_ui),
            asr_initial_prompt=initial_prompt_for_language(lang_ui),
            **kw,
        )

    def _start_wav_if_enabled(self) -> Optional[str]:
        if not self.chk_wav.isChecked():
            return None
        if not self._wav_recording_available():
            return "WAV disabled: install soundfile."
        out_path = self._current_output_path()
        self.output_name = out_path.name
        try:
            self.writer.start_recording(out_path, self.fmt)
        except Exception as e:
            return f"WAV start failed: {e}"
        return None

    def _post_start_audio_check(self) -> None:
        if not self._is_running():
            return
        srcs = self.engine.get_meters().get("sources", {})
        now_mono = time.monotonic()
        failed = self._collect_source_failures(srcs)
        silent = self._collect_silent_sources(srcs, now_mono)

        if failed:
            for msg in failed:
                self._append_transcript_line(f"[{self._fmt_ts(time.time())}] SOURCE ERROR: {msg}")
            self._set_status("Audio source failed. See transcript for details.")
            return

        if silent:
            joined = ", ".join(silent[:3])
            more = "..." if len(silent) > 3 else ""
            self._warn_throttle(
                f"No audio frames detected from: {joined}{more}. Try re-adding loopback device and Start again.",
                min_interval_s=0.1,
            )

    def _collect_source_failures(self, srcs: dict) -> List[str]:
        failed: List[str] = []
        for name in self.rows:
            src_obj = self.source_objs.get(name)
            if src_obj is None or not hasattr(src_obj, "get_last_error"):
                continue
            try:
                err = src_obj.get_last_error()
            except Exception:
                err = None
            if err:
                failed.append(f"{name}: {err}")
        return failed

    def _collect_silent_sources(self, srcs: dict, now_mono: float) -> List[str]:
        silent: List[str] = []
        for name, row in self.rows.items():
            if not row.enabled.isChecked():
                continue
            info = srcs.get(name, {})
            last_ts = float(info.get("last_ts", 0.0))
            rms = float(info.get("rms", 0.0))
            if last_ts <= 0.0 or ((now_mono - last_ts) > 2.0 and rms <= 1e-5):
                silent.append(name)
        return silent
