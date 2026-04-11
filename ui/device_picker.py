from __future__ import annotations

from typing import Tuple

from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QVBoxLayout, QWidget

from application.device_catalog import (
    LOOPBACK_SOURCE_TYPE,
    MIC_SOURCE_TYPE,
    list_input_devices,
    list_loopback_devices,
)


class DevicePickerDialog(QDialog):
    TYPE_LOOPBACK = LOOPBACK_SOURCE_TYPE
    TYPE_MIC = MIC_SOURCE_TYPE

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Add device")
        self.resize(720, 190)

        self.cmb_type = QComboBox()
        self.cmb_type.addItems([self.TYPE_LOOPBACK, self.TYPE_MIC])

        self.cmb_device = QComboBox()
        self.cmb_device.setEditable(False)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.addRow("Source type:", self.cmb_type)
        form.addRow("Device:", self.cmb_device)
        layout.addLayout(form)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(self.buttons)

        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        self.cmb_type.currentIndexChanged.connect(self._reload_devices)
        self._reload_devices()

    def _reload_devices(self) -> None:
        self.cmb_device.clear()
        source_type = self.cmb_type.currentText()

        if source_type == self.TYPE_LOOPBACK:
            devices = list_loopback_devices()
            empty_label = "(no loopback devices found)"
        else:
            devices = list_input_devices()
            empty_label = "(no input devices found)"

        if not devices:
            self.cmb_device.addItem(empty_label, None)
            self.cmb_device.setEnabled(False)
            return

        self.cmb_device.setEnabled(True)
        for label, token in devices:
            self.cmb_device.addItem(label, token)

    def selected(self) -> Tuple[str, object | None]:
        source_type = self.cmb_type.currentText()
        token = self.cmb_device.currentData()
        return source_type, token
