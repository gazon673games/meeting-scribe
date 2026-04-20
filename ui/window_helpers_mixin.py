from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtWidgets import QLabel, QLineEdit, QProgressBar

from application.event_types import SourceErrorEvent


class WindowHelpersMixin:
    def _set_status(self, text: str) -> None:
        self._set_label_text_if_changed(self.lbl_status, text)

    @staticmethod
    def _set_label_text_if_changed(label: QLabel, text: str) -> None:
        if label.text() != text:
            label.setText(text)

    @staticmethod
    def _set_progress_if_changed(bar: QProgressBar, value: int) -> None:
        if int(bar.value()) != int(value):
            bar.setValue(int(value))

    @staticmethod
    def _set_line_edit_if_changed(edit: QLineEdit, text: str) -> None:
        if edit.text() != text:
            edit.setText(text)

    def _on_source_error(self, source: str, error: str) -> None:
        ev = SourceErrorEvent(source=str(source), error=str(error), ts=time.time())
        try:
            self.asr_ui_q.put_nowait(ev)
        except Exception:
            pass

    def _is_running(self) -> bool:
        session_state = getattr(self, "_session_state", None)
        if session_state is not None:
            return bool(session_state.is_running)
        return self.engine.is_running()

    def _wav_recording_available(self) -> bool:
        return bool(self.wav_recorder_factory.available())

    def _offline_asr_available(self) -> bool:
        return bool(self.offline_pass_use_case.available())

    def _current_output_path(self) -> Path:
        name = (self.txt_output.text() or "").strip()
        if not name:
            name = "capture_mix.wav"
        name = Path(name).name
        if not name.lower().endswith(".wav"):
            name += ".wav"
        return self.project_root / name

    def _on_output_changed(self, _text: str) -> None:
        if self._is_running() and self.chk_wav.isChecked():
            self.txt_output.blockSignals(True)
            self.txt_output.setText(self.output_name)
            self.txt_output.blockSignals(False)
            return

        p = self._current_output_path()
        self.output_name = p.name
        if self.txt_output.text() != self.output_name:
            self.txt_output.blockSignals(True)
            self.txt_output.setText(self.output_name)
            self.txt_output.blockSignals(False)

    @staticmethod
    def _safe_int(s: str, default: int, lo: int, hi: int) -> int:
        try:
            v = int(str(s).strip())
        except Exception:
            v = int(default)
        v = max(int(lo), min(int(hi), int(v)))
        return int(v)

    @staticmethod
    def _safe_float(s: str, default: float, lo: float, hi: float) -> float:
        try:
            v = float(str(s).strip().replace(",", "."))
        except Exception:
            v = float(default)
        v = max(float(lo), min(float(hi), float(v)))
        return float(v)

    def closeEvent(self, event) -> None:
        self._closing = True
        try:
            self._stop_all(run_offline_pass=False, wait=True)
        finally:
            self._stop_codex_timer()
            try:
                self.writer.stop()
            except Exception:
                pass
            self._rt_close()
            self._human_log_close()
        event.accept()
