from __future__ import annotations

import time
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar

from ui.device_picker import DevicePickerDialog


@dataclass
class SourceRow:
    name: str
    enabled: QCheckBox
    meter: QProgressBar
    status: QLabel
    delay_ms: QLineEdit


class SourceControlsMixin:
    def _make_unique_name(self, base: str) -> str:
        if base not in self.rows:
            return base
        i = 2
        while f"{base}_{i}" in self.rows:
            i += 1
        return f"{base}_{i}"

    def _add_row(self, name: str) -> None:
        if name in self.rows:
            return

        row_layout = QHBoxLayout()

        checkbox = QCheckBox(name)
        checkbox.setTristate(False)
        checkbox.setChecked(True)

        delay = QLineEdit("0")
        delay.setFixedWidth(70)
        delay.setPlaceholderText("ms")
        delay.setToolTip("Delay compensation in milliseconds (>=0).")

        meter = QProgressBar()
        meter.setRange(0, 100)
        meter.setTextVisible(False)

        status = QLabel("idle")
        status.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        checkbox.clicked.connect(lambda checked, n=name: self._on_source_toggle(n, checked))
        delay.editingFinished.connect(lambda n=name: self._apply_delay_from_ui(n))

        row_layout.addWidget(checkbox, 0)
        row_layout.addWidget(QLabel("delay ms:"), 0)
        row_layout.addWidget(delay, 0)
        row_layout.addWidget(meter, 1)
        row_layout.addWidget(status, 0)

        self.grp_layout.addLayout(row_layout)
        self.rows[name] = SourceRow(name=name, enabled=checkbox, meter=meter, status=status, delay_ms=delay)

    def _apply_delay_from_ui(self, name: str) -> None:
        row = self.rows.get(name)
        if row is None:
            return
        text = (row.delay_ms.text() or "").strip().replace(",", ".")
        try:
            value = float(text) if text else 0.0
        except ValueError:
            value = 0.0
        if value < 0:
            value = 0.0
        row.delay_ms.setText(str(int(round(value))) if abs(value - round(value)) < 1e-6 else f"{value:.2f}")
        self.engine.set_source_delay_ms(name, value)

    def _on_source_toggle(self, name: str, checked: bool) -> None:
        self.engine.set_source_enabled(name, bool(checked))
        self._append_transcript_line(
            f"[{self._fmt_ts(time.time())}] UI toggle: {name} -> {'ON' if checked else 'MUTED'}"
        )

        if self._is_running() and self.chk_asr.isChecked() and self.cmb_asr_mode.currentIndex() == 1:
            enabled_sources = [n for n, row in self.rows.items() if row.enabled.isChecked()]
            self.engine.set_tap_config(mode="sources", sources=enabled_sources, drop_threshold=0.85)

    def _add_device_dialog(self) -> None:
        if self._is_running():
            self._set_status("Stop before adding devices.")
            return

        dlg = DevicePickerDialog(self, catalog=self.device_catalog)
        if dlg.exec() != DevicePickerDialog.Accepted:
            return

        source_type, token = dlg.selected()
        if token is None:
            self._set_status("No device selected.")
            return

        try:
            if source_type == DevicePickerDialog.TYPE_LOOPBACK:
                name = self._make_unique_name("desktop_audio")
                source = self.audio_source_factory.create_loopback_source(
                    name=name,
                    engine_format=self.fmt,
                    device=token,
                    error_callback=self._on_source_error,
                )
                self.engine.add_source(source)
                self.source_objs[name] = source
                self._add_row(name)
                self._set_status(f"Added loopback -> source '{name}'")
            else:
                name = self._make_unique_name("mic")
                source = self.audio_source_factory.create_microphone_source(name=name, device=token)
                self.engine.add_source(source)
                self.source_objs[name] = source
                self._add_row(name)
                self._set_status(f"Added mic -> source '{name}'")
        except Exception as e:
            self._set_status(f"Failed to add device: {e}")
