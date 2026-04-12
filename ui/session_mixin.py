from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QTimer

from application.asr_language import initial_prompt_for_language, normalize_asr_language, runtime_asr_language
from application.asr_session import ASRRuntime, ASRSessionSettings
from application.offline_pass import offline_asr_available, run_offline_asr_pass
from infrastructure.wav_recording import wav_recording_available


class SessionMixin:
    def _start_all(self) -> None:
        if self._is_running():
            return
        if self._asr_stop_active:
            self._set_status("ASR is still stopping. Wait for it to finish first.")
            return
        if self._offline_pass_active:
            self._set_status("Offline pass is still running. Wait for it to finish first.")
            return
        if len(self.rows) == 0:
            self._set_status("Add at least one device first.")
            return

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

        if self.chk_asr.isChecked():
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
            self._set_status(f"Engine start failed: {e}")
            return

        QTimer.singleShot(1400, self._post_start_audio_check)

        self.asr_running = False
        self.asr = None
        self._start_asr_if_enabled()
        self._start_wav_if_enabled()
        self._set_session_controls_running()

        self.ui_timer.start()
        self._set_status(f"running: ASR={'on' if self.asr_running else 'off'}, WAV={'on' if self.writer.is_recording() else 'off'}")

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

        try:
            self.asr = self.asr_runtime_factory.build(
                settings,
                tap_queue=self.tap_q,
                project_root=self.project_root,
                event_queue=self.asr_ui_q,
            )
            self.asr.start()
            self.asr_running = True

            human_log_path = self._human_log_open_session()
            if human_log_path is not None:
                self._append_transcript_line(f"[{self._fmt_ts(time.time())}] human log -> {human_log_path}")
        except Exception as e:
            self._set_status(f"ASR start failed: {e}")

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
        if not wav_recording_available():
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
        self._asr_stop_active = False
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

        self.chk_offline_on_stop.setEnabled(offline_asr_available())
        self.chk_rt_transcript_file.setEnabled(True)

        self._set_custom_enabled(self.cmb_profile.currentText() == self.PROFILE_CUSTOM)

        self.chk_wav.setEnabled(wav_recording_available())
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
            and offline_asr_available()
            and bool(self.chk_offline_on_stop.isChecked())
            and wav_recording_available()
            and Path(wav_path).exists()
        ):
            self._start_offline_pass(
                Path(wav_path),
                model_name=offline_model_name,
                language=offline_language,
            )

    def _run_asr_stop_worker(
        self,
        asr_obj: ASRRuntime,
        *,
        wav_path: Path,
        run_offline_pass: bool,
        offline_model_name: str,
        offline_language: Optional[str],
    ) -> None:
        stop_error: Optional[str] = None
        try:
            asr_obj.stop()
        except Exception as e:
            stop_error = f"{type(e).__name__}: {e}"

        self.background_event.emit(
            {
                "type": "asr_stop_done",
                "wav_path": str(wav_path),
                "run_offline_pass": bool(run_offline_pass),
                "offline_model_name": str(offline_model_name),
                "offline_language": offline_language,
                "stop_error": stop_error,
            }
        )

    def _stop_all(self, *, run_offline_pass: bool = True, wait: bool = False) -> None:
        if self._asr_stop_active:
            if wait and self._asr_stop_thread is not None:
                self._asr_stop_thread.join()
            else:
                self._set_status("ASR is still stopping. Wait for it to finish.")
            return

        wav_path = self.writer.target_path() or self._current_output_path()
        offline_model_name = str(self.cmb_model.currentText() or "large-v3")
        lang_ui = normalize_asr_language(self.cmb_lang.currentText())
        offline_language = runtime_asr_language(lang_ui)

        if self.writer.is_recording():
            self.writer.stop_recording()

        asr_to_stop = self.asr
        self.asr = None
        self.asr_running = False

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
                run_offline_pass=run_offline_pass,
                offline_model_name=offline_model_name,
                offline_language=offline_language,
            )
            return

        self._asr_stop_active = True
        self._set_status("stopping (waiting for ASR to finish current transcription)...")

        if wait or self._closing:
            stop_error: Optional[str] = None
            try:
                asr_to_stop.stop()
            except Exception as e:
                stop_error = f"{type(e).__name__}: {e}"
            self._finish_stop_ui(
                wav_path=Path(wav_path),
                run_offline_pass=run_offline_pass,
                offline_model_name=offline_model_name,
                offline_language=offline_language,
                stop_error=stop_error,
            )
            return

        self._asr_stop_thread = threading.Thread(
            target=self._run_asr_stop_worker,
            kwargs={
                "asr_obj": asr_to_stop,
                "wav_path": Path(wav_path),
                "run_offline_pass": run_offline_pass,
                "offline_model_name": offline_model_name,
                "offline_language": offline_language,
            },
            name="asr-stop-worker",
            daemon=True,
        )
        self._asr_stop_thread.start()

    def _start_offline_pass(self, wav_path: Path, *, model_name: str, language: Optional[str]) -> None:
        if self._offline_pass_active:
            self._set_status("Offline pass is already running.")
            return
        if not offline_asr_available():
            self._set_status("Offline pass is unavailable.")
            return

        self._offline_pass_active = True
        self.btn_start.setEnabled(False)

        self._offline_thread = threading.Thread(
            target=self._run_offline_pass_worker,
            args=(Path(wav_path), str(model_name), language),
            name="offline-asr-pass",
            daemon=True,
        )
        self._offline_thread.start()

    def _run_offline_pass_worker(self, wav_path: Path, model_name: str, language: Optional[str]) -> None:
        try:
            self.background_event.emit({"type": "offline_pass_started"})

            logs_dir = self.project_root / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_txt = logs_dir / f"offline_transcript_{ts}.txt"

            result_path = run_offline_asr_pass(
                project_root=self.project_root,
                wav_path=Path(wav_path),
                out_txt=out_txt,
                model_name=str(model_name or "large-v3"),
                language=language,
            )

            self.background_event.emit(
                {
                    "type": "offline_pass_done",
                    "out_txt": str(result_path),
                }
            )
        except Exception as e:
            self.background_event.emit(
                {
                    "type": "offline_pass_error",
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    def _handle_background_event(self, ev: Dict[str, Any]) -> None:
        if self._closing:
            return

        typ = str(ev.get("type", ""))
        if typ == "asr_stop_done":
            stop_error_raw = ev.get("stop_error")
            self._finish_stop_ui(
                wav_path=Path(str(ev.get("wav_path", self._current_output_path()))),
                run_offline_pass=bool(ev.get("run_offline_pass", False)),
                offline_model_name=str(ev.get("offline_model_name", self.cmb_model.currentText() or "large-v3")),
                offline_language=ev.get("offline_language"),
                stop_error=(str(stop_error_raw).strip() if stop_error_raw is not None else None),
            )
            return

        if typ == "offline_pass_started":
            self._append_transcript_line(f"[{self._fmt_ts(time.time())}] offline pass: starting...")
            self._set_status("offline pass: running")
            return

        if typ == "offline_pass_done":
            self._offline_pass_active = False
            self._offline_thread = None
            self.btn_start.setEnabled(True)
            out_txt = str(ev.get("out_txt", "")).strip()
            self._append_transcript_line(f"[{self._fmt_ts(time.time())}] offline pass: done -> {out_txt}")
            self._set_status("offline pass: done")
            return

        if typ == "offline_pass_error":
            self._offline_pass_active = False
            self._offline_thread = None
            self.btn_start.setEnabled(True)
            err = str(ev.get("error", "unknown error"))
            self._append_transcript_line(f"[{self._fmt_ts(time.time())}] offline pass: ERROR {err}")
            self._set_status("offline pass: failed")
            return
