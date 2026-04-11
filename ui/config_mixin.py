from __future__ import annotations

import json
from typing import Any, Dict

from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit

from application.asr_language import SUPPORTED_ASR_LANGUAGES
from application.asr_profiles import profile_defaults
from application.recording import wav_recording_available

CONFIG_VERSION = 2


class MainWindowConfigMixin:
    def _wire_config_change(self, w) -> None:
        try:
            if isinstance(w, QLineEdit):
                w.textChanged.connect(lambda _t: self._mark_config_dirty())
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(lambda _s: self._mark_config_dirty())
            elif isinstance(w, QComboBox):
                w.currentIndexChanged.connect(lambda _i: self._mark_config_dirty())
        except Exception:
            pass

    def _mark_config_dirty(self) -> None:
        self._cfg_dirty = True
        if not self._cfg_save_timer.isActive():
            self._cfg_save_timer.start()

    def _flush_config_if_dirty(self) -> None:
        if not self._cfg_dirty:
            self._cfg_save_timer.stop()
            return
        self._cfg_dirty = False
        self._cfg_save_timer.stop()
        try:
            cfg = self._build_config_from_ui()
            tmp = self.config_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.config_path)
        except Exception:
            pass

    def _build_config_from_ui(self) -> Dict[str, Any]:
        return {
            "version": CONFIG_VERSION,
            "ui": {
                "asr_enabled": bool(self.chk_asr.isChecked()),
                "lang": str(self.cmb_lang.currentText()),
                "asr_mode": int(self.cmb_asr_mode.currentIndex()),
                "model": str(self.cmb_model.currentText()),
                "profile": str(self.cmb_profile.currentText()),
                "wav_enabled": bool(self.chk_wav.isChecked()),
                "output_file": str(self.txt_output.text() or "").strip(),
                "long_run": bool(self.chk_longrun.isChecked()),
                "rt_transcript_to_file": bool(self.chk_rt_transcript_file.isChecked()),
                "offline_on_stop": bool(self.chk_offline_on_stop.isChecked()),
                "asr_settings_expanded": bool(self.btn_asr_toggle.isChecked()),
            },
            "asr": {
                "compute_type": str(self.cmb_compute.currentText()),
                "beam_size": self._safe_int(self.txt_beam.text(), 5, 1, 20),
                "endpoint_silence_ms": self._safe_float(self.txt_endpoint.text(), 650.0, 50.0, 5000.0),
                "max_segment_s": self._safe_float(self.txt_maxseg.text(), 7.0, 1.0, 60.0),
                "overlap_ms": self._safe_float(self.txt_overlap.text(), 200.0, 0.0, 2000.0),
                "vad_energy_threshold": self._safe_float(self.txt_vad_thr.text(), 0.0055, 1e-5, 1.0),
                "overload_strategy": str(self.cmb_overload_strategy.currentText()),
                "overload_enter_qsize": self._safe_int(self.txt_over_enter.text(), 18, 1, 999),
                "overload_exit_qsize": self._safe_int(self.txt_over_exit.text(), 6, 1, 999),
                "overload_hard_qsize": self._safe_int(self.txt_over_hard.text(), 28, 1, 999),
                "overload_beam_cap": self._safe_int(self.txt_over_beamcap.text(), 2, 1, 20),
                "overload_max_segment_s": self._safe_float(self.txt_over_maxseg.text(), 5.0, 0.5, 60.0),
                "overload_overlap_ms": self._safe_float(self.txt_over_overlap.text(), 120.0, 0.0, 2000.0),
            },
            "codex": self._build_codex_config(),
        }

    def _load_config_into_ui(self) -> None:
        if not self.config_path.exists():
            return
        try:
            cfg = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception:
            return

        ui = cfg.get("ui", {}) if isinstance(cfg, dict) else {}
        asr = cfg.get("asr", {}) if isinstance(cfg, dict) else {}
        codex = cfg.get("codex", {}) if isinstance(cfg, dict) else {}

        try:
            if "asr_enabled" in ui:
                self.chk_asr.setChecked(bool(ui.get("asr_enabled")))
            if "lang" in ui and str(ui.get("lang")) in SUPPORTED_ASR_LANGUAGES:
                self.cmb_lang.setCurrentText(str(ui.get("lang")))
            if "asr_mode" in ui:
                idx = int(ui.get("asr_mode", 1))
                self.cmb_asr_mode.setCurrentIndex(1 if idx == 1 else 0)
            if "model" in ui and str(ui.get("model")) in ("large-v3", "medium", "small"):
                self.cmb_model.setCurrentText(str(ui.get("model")))
            if "profile" in ui and str(ui.get("profile")) in (
                self.PROFILE_REALTIME,
                self.PROFILE_BALANCED,
                self.PROFILE_QUALITY,
                self.PROFILE_CUSTOM,
            ):
                self.cmb_profile.setCurrentText(str(ui.get("profile")))
            if "wav_enabled" in ui and wav_recording_available():
                self.chk_wav.setChecked(bool(ui.get("wav_enabled")))
            if "output_file" in ui:
                val = str(ui.get("output_file") or "").strip()
                if val:
                    self.txt_output.setText(val)
            if "long_run" in ui:
                self.chk_longrun.setChecked(bool(ui.get("long_run")))
            if "rt_transcript_to_file" in ui:
                self.chk_rt_transcript_file.setChecked(bool(ui.get("rt_transcript_to_file")))
            if "offline_on_stop" in ui and self.chk_offline_on_stop.isEnabled():
                self.chk_offline_on_stop.setChecked(bool(ui.get("offline_on_stop")))
            if "asr_settings_expanded" in ui:
                expanded = bool(ui.get("asr_settings_expanded"))
                self.btn_asr_toggle.setChecked(expanded)
                self._apply_asr_settings_visibility(expanded=expanded)
        except Exception:
            pass

        try:
            if "compute_type" in asr:
                v = str(asr.get("compute_type"))
                if v:
                    self.cmb_compute.setCurrentText(v)
            if "beam_size" in asr:
                self.txt_beam.setText(str(int(asr.get("beam_size"))))
            if "endpoint_silence_ms" in asr:
                self.txt_endpoint.setText(str(float(asr.get("endpoint_silence_ms"))))
            if "max_segment_s" in asr:
                self.txt_maxseg.setText(str(float(asr.get("max_segment_s"))))
            if "overlap_ms" in asr:
                self.txt_overlap.setText(str(float(asr.get("overlap_ms"))))
            if "vad_energy_threshold" in asr:
                self.txt_vad_thr.setText(str(float(asr.get("vad_energy_threshold"))))
            if "overload_strategy" in asr:
                v = str(asr.get("overload_strategy")).strip().lower()
                self.cmb_overload_strategy.setCurrentText("keep_all" if v == "keep_all" else "drop_old")
            if "overload_enter_qsize" in asr:
                self.txt_over_enter.setText(str(int(asr.get("overload_enter_qsize"))))
            if "overload_exit_qsize" in asr:
                self.txt_over_exit.setText(str(int(asr.get("overload_exit_qsize"))))
            if "overload_hard_qsize" in asr:
                self.txt_over_hard.setText(str(int(asr.get("overload_hard_qsize"))))
            if "overload_beam_cap" in asr:
                self.txt_over_beamcap.setText(str(int(asr.get("overload_beam_cap"))))
            if "overload_max_segment_s" in asr:
                self.txt_over_maxseg.setText(str(float(asr.get("overload_max_segment_s"))))
            if "overload_overlap_ms" in asr:
                self.txt_over_overlap.setText(str(float(asr.get("overload_overlap_ms"))))
        except Exception:
            pass

        self._load_codex_from_config(codex)
        self._on_longrun_changed()

    def _apply_profile_to_fields(self, profile: str, *, force: bool = False) -> None:
        if (profile or "") != self.PROFILE_CUSTOM:
            defaults = profile_defaults(profile)
            self.cmb_compute.setCurrentText(str(defaults["compute_type"]))
            self.txt_beam.setText(str(int(defaults["beam_size"])))
            self.txt_endpoint.setText(str(float(defaults["endpoint_silence_ms"])))
            self.txt_maxseg.setText(str(float(defaults["max_segment_s"])))
            self.txt_overlap.setText(str(float(defaults["overlap_ms"])))
            self.txt_vad_thr.setText(str(float(defaults["vad_energy_threshold"])))

            self.cmb_overload_strategy.setCurrentText(str(defaults["overload_strategy"]))
            self.txt_over_enter.setText(str(int(defaults["overload_enter_qsize"])))
            self.txt_over_exit.setText(str(int(defaults["overload_exit_qsize"])))
            self.txt_over_hard.setText(str(int(defaults["overload_hard_qsize"])))
            self.txt_over_beamcap.setText(str(int(defaults["overload_beam_cap"])))
            self.txt_over_maxseg.setText(str(float(defaults["overload_max_segment_s"])))
            self.txt_over_overlap.setText(str(float(defaults["overload_overlap_ms"])))

            self._set_custom_enabled(False)
            self._mark_config_dirty()
            return

        self._set_custom_enabled(True)
        if force:
            self._mark_config_dirty()

    def _set_custom_enabled(self, enabled: bool) -> None:
        for widget in [
            self.cmb_compute,
            self.txt_beam,
            self.txt_endpoint,
            self.txt_maxseg,
            self.txt_overlap,
            self.txt_vad_thr,
            self.cmb_overload_strategy,
            self.txt_over_enter,
            self.txt_over_exit,
            self.txt_over_hard,
            self.txt_over_beamcap,
            self.txt_over_maxseg,
            self.txt_over_overlap,
        ]:
            try:
                widget.setEnabled(bool(enabled))
            except Exception:
                pass

    def _on_profile_changed(self) -> None:
        profile = self.cmb_profile.currentText()
        self._apply_profile_to_fields(profile, force=True)

    def _apply_asr_settings_visibility(self, *, expanded: bool) -> None:
        self.grp_asr_cfg.setVisible(bool(expanded))
        self.btn_asr_toggle.setText("Hide ASR settings" if expanded else "Show ASR settings")

    def _toggle_asr_settings(self) -> None:
        expanded = bool(self.btn_asr_toggle.isChecked())
        self._apply_asr_settings_visibility(expanded=expanded)
        self._mark_config_dirty()

    def _on_longrun_changed(self) -> None:
        self._long_run_mode = bool(self.chk_longrun.isChecked())
        interval = self._ui_interval_long_ms if self._long_run_mode else self._ui_interval_normal_ms
        self.ui_timer.setInterval(int(interval))
        self._mark_config_dirty()
