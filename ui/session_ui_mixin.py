from __future__ import annotations

from enum import Enum


class SessionUiMode(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"


class SessionUiMixin:
    def _set_session_controls_running(self) -> None:
        self._apply_session_ui_mode(SessionUiMode.RUNNING)

    def _set_stop_ui_pending(self) -> None:
        self._apply_session_ui_mode(SessionUiMode.STOPPING)

    def _apply_session_ui_mode(self, mode: SessionUiMode) -> None:
        idle = mode == SessionUiMode.IDLE
        running = mode == SessionUiMode.RUNNING

        self.btn_start.setEnabled(idle)
        self.btn_stop.setEnabled(running)

        for widget in [
            self.btn_add,
            self.chk_asr,
            self.cmb_profile,
            self.cmb_lang,
            self.cmb_asr_mode,
            self.cmb_model,
            self.grp_asr_cfg,
            self.btn_asr_toggle,
            self.chk_longrun,
        ]:
            widget.setEnabled(idle)

        self.chk_offline_on_stop.setEnabled(idle and self._offline_asr_available())
        self.chk_rt_transcript_file.setEnabled(idle)
        self.chk_wav.setEnabled(idle and self._wav_recording_available())
        self.txt_output.setEnabled(idle)

        if idle:
            self._set_custom_enabled(self.cmb_profile.currentText() == self.PROFILE_CUSTOM)
