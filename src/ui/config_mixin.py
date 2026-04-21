from __future__ import annotations

from typing import Any, Dict

from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit

from application.asr_language import SUPPORTED_ASR_LANGUAGES
from application.asr_profiles import profile_defaults
from application.commands import SwitchProfileCommand
from application.model_policy import ASR_MODEL_NAMES, ModelOrchestrator
from ui.asr_field_defs import (
    _ASR_ALL_FIELDS,
    _ASR_COMBO_FIELDS,
    _ASR_CUSTOM_WIDGET_ATTRS,
    _ASR_FLOAT_FIELDS,
    _ASR_INT_FIELDS,
)

CONFIG_VERSION = 2


class MainWindowConfigMixin:
    # ── dirty tracking ─────────────────────────────────────────────────

    def _wire_config_change(self, w) -> None:
        try:
            if isinstance(w, QLineEdit):
                w.textChanged.connect(lambda _: self._mark_config_dirty())
            elif isinstance(w, QCheckBox):
                w.stateChanged.connect(lambda _: self._mark_config_dirty())
            elif isinstance(w, QComboBox):
                w.currentIndexChanged.connect(lambda _: self._mark_config_dirty())
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
            self.config_repository.write(self._build_config_from_ui())
        except Exception:
            pass

    # ── config ↔ UI ────────────────────────────────────────────────────

    def _build_config_from_ui(self) -> Dict[str, Any]:
        return {
            "version": CONFIG_VERSION,
            "ui": self._read_ui_section(),
            "asr": self._read_asr_section(),
            "codex": self._build_codex_config(),
        }

    def _read_ui_section(self) -> Dict[str, Any]:
        return {
            "asr_enabled":           bool(self.chk_asr.isChecked()),
            "lang":                  str(self.cmb_lang.currentText()),
            "asr_mode":              int(self.cmb_asr_mode.currentIndex()),
            "model":                 str(self.cmb_model.currentText()),
            "profile":               str(self.cmb_profile.currentText()),
            "wav_enabled":           bool(self.chk_wav.isChecked()),
            "output_file":           str(self.txt_output.text() or "").strip(),
            "long_run":              bool(self.chk_longrun.isChecked()),
            "rt_transcript_to_file": bool(self.chk_rt_transcript_file.isChecked()),
            "offline_on_stop":       bool(self.chk_offline_on_stop.isChecked()),
            "asr_settings_expanded": bool(self.btn_asr_toggle.isChecked()),
        }

    def _read_asr_section(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key, widget, default, lo, hi in _ASR_INT_FIELDS:
            data[key] = self._safe_int(getattr(self, widget).text(), default, lo, hi)
        for key, widget, default, lo, hi in _ASR_FLOAT_FIELDS:
            data[key] = self._safe_float(getattr(self, widget).text(), default, lo, hi)
        for key, widget in _ASR_COMBO_FIELDS:
            data[key] = getattr(self, widget).currentText()
        return data

    def _load_config_into_ui(self) -> None:
        if not self.config_repository.exists():
            return
        try:
            cfg = self.config_repository.read()
        except Exception:
            return
        ui    = cfg.get("ui",    {}) if isinstance(cfg, dict) else {}
        asr   = cfg.get("asr",   {}) if isinstance(cfg, dict) else {}
        codex = cfg.get("codex", {}) if isinstance(cfg, dict) else {}
        try:
            self._write_ui_section(ui)
        except Exception:
            pass
        try:
            self._write_asr_section(asr)
        except Exception:
            pass
        self._load_codex_from_config(codex)
        self._on_longrun_changed()

    def _write_ui_section(self, ui: dict) -> None:
        if "asr_enabled" in ui:
            self.chk_asr.setChecked(bool(ui["asr_enabled"]))
        if "lang" in ui and str(ui["lang"]) in SUPPORTED_ASR_LANGUAGES:
            self.cmb_lang.setCurrentText(str(ui["lang"]))
        if "asr_mode" in ui:
            self.cmb_asr_mode.setCurrentIndex(1 if int(ui["asr_mode"]) == 1 else 0)
        if "model" in ui and str(ui["model"]) in ASR_MODEL_NAMES:
            self.cmb_model.setCurrentText(str(ui["model"]))
        if "profile" in ui and str(ui["profile"]) in self._valid_profiles():
            self.cmb_profile.setCurrentText(str(ui["profile"]))
        if "wav_enabled" in ui and self._wav_recording_available():
            self.chk_wav.setChecked(bool(ui["wav_enabled"]))
        if "output_file" in ui:
            val = str(ui["output_file"] or "").strip()
            if val:
                self.txt_output.setText(val)
        if "long_run" in ui:
            self.chk_longrun.setChecked(bool(ui["long_run"]))
        if "rt_transcript_to_file" in ui:
            self.chk_rt_transcript_file.setChecked(bool(ui["rt_transcript_to_file"]))
        if "offline_on_stop" in ui and self.chk_offline_on_stop.isEnabled():
            self.chk_offline_on_stop.setChecked(bool(ui["offline_on_stop"]))
        if "asr_settings_expanded" in ui:
            expanded = bool(ui["asr_settings_expanded"])
            self.btn_asr_toggle.setChecked(expanded)
            self._apply_asr_settings_visibility(expanded=expanded)

    def _write_asr_section(self, asr: dict) -> None:
        for key, widget, default, *_ in _ASR_ALL_FIELDS:
            if key in asr:
                cast = int if isinstance(default, int) else float
                getattr(self, widget).setText(str(cast(asr[key])))
        for key, widget in _ASR_COMBO_FIELDS:
            if key in asr:
                v = str(asr[key])
                if key == "overload_strategy":
                    v = "keep_all" if v.strip().lower() == "keep_all" else "drop_old"
                if v:
                    getattr(self, widget).setCurrentText(v)

    # ── profile ────────────────────────────────────────────────────────

    def _apply_profile_to_fields(self, profile: str, *, force: bool = False) -> None:
        if (profile or "") == self.PROFILE_CUSTOM:
            self._set_custom_enabled(True)
            if force:
                self._mark_config_dirty()
            return
        self._write_asr_section(profile_defaults(profile))
        self._apply_model_policy_to_ui(profile, mark_dirty=force)
        self._set_custom_enabled(False)
        self._mark_config_dirty()

    def _set_custom_enabled(self, enabled: bool) -> None:
        for attr in _ASR_CUSTOM_WIDGET_ATTRS:
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    widget.setEnabled(bool(enabled))
                except Exception:
                    pass

    def _on_profile_changed(self) -> None:
        command = SwitchProfileCommand(profile=self.cmb_profile.currentText())
        dispatcher = getattr(self, "_command_dispatcher", None)
        if dispatcher is None:
            self._handle_switch_profile_command(command)
            return
        try:
            dispatcher.dispatch(command)
        except KeyError:
            dispatcher.register(SwitchProfileCommand, self._handle_switch_profile_command)
            dispatcher.dispatch(command)

    def _handle_switch_profile_command(self, command: SwitchProfileCommand) -> None:
        self._apply_profile_to_fields(command.profile, force=True)

    def _on_policy_input_changed(self) -> None:
        profile = self.cmb_profile.currentText()
        if (profile or "") == self.PROFILE_CUSTOM:
            self._mark_config_dirty()
            return
        self._apply_model_policy_to_ui(profile, mark_dirty=True)
        self._mark_config_dirty()

    def _apply_model_policy_to_ui(self, profile: str, *, mark_dirty: bool) -> None:
        available = [self.cmb_model.itemText(i) for i in range(self.cmb_model.count())]
        decision = ModelOrchestrator().recommend(
            asr_profile=profile,
            language=str(self.cmb_lang.currentText()),
            current_asr_model=str(self.cmb_model.currentText()),
            available_asr_models=available,
            codex_profiles=list(getattr(self, "_codex_profiles", [])),
            current_codex_profile_id=str(getattr(self, "_codex_selected_profile_id", "")),
        )
        if decision.asr_model and decision.asr_model != self.cmb_model.currentText():
            self.cmb_model.setCurrentText(decision.asr_model)
        if decision.codex_profile_id and hasattr(self, "_set_codex_selected_profile"):
            self._set_codex_selected_profile(decision.codex_profile_id, mark_dirty=mark_dirty)

    # ── visibility / long-run ──────────────────────────────────────────

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

    def _valid_profiles(self) -> tuple:
        return (self.PROFILE_REALTIME, self.PROFILE_BALANCED, self.PROFILE_QUALITY, self.PROFILE_CUSTOM)
