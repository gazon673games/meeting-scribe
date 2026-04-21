from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
)


class CodexPanelMixin:
    def _build_codex_header(self, root: QVBoxLayout) -> None:
        codex_hdr = QHBoxLayout()
        self.btn_codex_toggle = QPushButton("Show Codex helper")
        self.btn_codex_toggle.setCheckable(True)
        self.btn_codex_toggle.setChecked(False)
        self.btn_codex_toggle.setVisible(False)
        codex_hdr.addWidget(self.btn_codex_toggle)
        codex_hdr.addStretch(1)
        root.addLayout(codex_hdr)

    def _build_codex_panel(self, splitter: QSplitter) -> None:
        self.grp_codex = QGroupBox("Codex helper console")
        codex_layout = QVBoxLayout(self.grp_codex)

        self.lbl_codex_status = QLabel("idle")
        self.lbl_codex_status.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        codex_layout.addWidget(self.lbl_codex_status)

        self.codex_profiles_row = QHBoxLayout()
        codex_layout.addLayout(self.codex_profiles_row)
        self._build_codex_context_row(codex_layout)
        self._build_codex_action_row(codex_layout)
        self._build_codex_output(codex_layout)
        self._build_codex_input(codex_layout)

        self.grp_codex.setVisible(False)
        splitter.addWidget(self.grp_codex)
        splitter.setSizes([820, 0])

    def _build_codex_context_row(self, codex_layout: QVBoxLayout) -> None:
        context_row = QHBoxLayout()
        context_row.addWidget(QLabel("Context:"))
        self.cmb_codex_context = QComboBox()
        context_row.addWidget(self.cmb_codex_context, 1)
        self.btn_codex_context_refresh = QPushButton("Refresh")
        context_row.addWidget(self.btn_codex_context_refresh, 0)
        codex_layout.addLayout(context_row)
        self._refresh_codex_context_sources()

    def _build_codex_action_row(self, codex_layout: QVBoxLayout) -> None:
        codex_actions_row = QHBoxLayout()
        self.btn_codex_answer = QPushButton("Answer")
        self.btn_codex_answer.setToolTip("Fast answer from recent context")
        codex_actions_row.addWidget(self.btn_codex_answer, 0)

        self.btn_codex_summary = QPushButton("Summary")
        self.btn_codex_summary.setToolTip("Deep summary from full context")
        codex_actions_row.addWidget(self.btn_codex_summary, 0)
        codex_actions_row.addStretch(1)
        codex_layout.addLayout(codex_actions_row)

    def _build_codex_output(self, codex_layout: QVBoxLayout) -> None:
        self.txt_codex = QTextEdit()
        self.txt_codex.setReadOnly(True)
        self.txt_codex.setPlaceholderText("Codex output will appear here")
        self.txt_codex.setLineWrapMode(QTextEdit.WidgetWidth)
        self.txt_codex.document().setMaximumBlockCount(1200)
        self.txt_codex.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        codex_layout.addWidget(self.txt_codex, 1)

    def _build_codex_input(self, codex_layout: QVBoxLayout) -> None:
        codex_input_row = QHBoxLayout()
        self.txt_codex_input = QLineEdit()
        self.txt_codex_input.setPlaceholderText("Type request or ANSWER")
        codex_input_row.addWidget(self.txt_codex_input, 1)
        self.btn_codex_send = QPushButton("Send")
        codex_input_row.addWidget(self.btn_codex_send, 0)
        codex_layout.addLayout(codex_input_row)

    def _connect_codex_signals(self) -> None:
        self.btn_codex_toggle.clicked.connect(self._toggle_codex_console)
        self.btn_codex_answer.clicked.connect(self._on_codex_answer_clicked)
        self.btn_codex_summary.clicked.connect(self._on_codex_summary_clicked)
        self.btn_codex_context_refresh.clicked.connect(self._refresh_codex_context_sources)
        self.cmb_codex_context.currentIndexChanged.connect(self._on_codex_context_changed)
        self.btn_codex_send.clicked.connect(self._on_codex_send_clicked)
        self.txt_codex_input.returnPressed.connect(self._on_codex_send_clicked)

    def _set_codex_inputs_enabled(self, enabled: bool) -> None:
        self.txt_codex_input.setEnabled(bool(enabled))
        self.btn_codex_send.setEnabled(bool(enabled))
        self.btn_codex_answer.setEnabled(bool(enabled))
        self.btn_codex_summary.setEnabled(bool(enabled))
        self.cmb_codex_context.setEnabled(bool(enabled))
        self.btn_codex_context_refresh.setEnabled(bool(enabled))

    def _start_codex_timer(self) -> None:
        self.codex_timer = QTimer(self)
        self.codex_timer.setInterval(140)
        self.codex_timer.timeout.connect(lambda: self._drain_codex_ui_events(limit=10))
        self.codex_timer.start()

    def _stop_codex_timer(self) -> None:
        try:
            if self.codex_timer is not None and self.codex_timer.isActive():
                self.codex_timer.stop()
        except Exception:
            pass

    def _apply_codex_console_visibility(self, *, expanded: bool) -> None:
        show = bool(self._codex_enabled and expanded)
        self.grp_codex.setVisible(show)
        self.btn_codex_toggle.setText("Hide Codex helper" if show else "Show Codex helper")
        if show:
            self.splitter.setSizes([620, 260])
        else:
            self.splitter.setSizes([860, 0])

    def _toggle_codex_console(self) -> None:
        expanded = bool(self.btn_codex_toggle.isChecked())
        self._apply_codex_console_visibility(expanded=expanded)
        self._mark_config_dirty()

    def _append_codex_line(self, line: str) -> None:
        max_chars = 180_000
        if self.txt_codex.document().characterCount() > max_chars:
            self.txt_codex.clear()
            self.txt_codex.append("[codex console cleared: too large]")

        self.txt_codex.append(str(line))
        self.txt_codex.moveCursor(QTextCursor.End)
        self.txt_codex.ensureCursorVisible()
