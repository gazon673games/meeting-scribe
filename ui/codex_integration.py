from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
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

from application.codex_assistant import CodexExecutionSettings
from application.codex_config import (
    CodexProfile,
    CodexSettings,
    codex_settings_to_dict,
    parse_codex_settings,
)
from application.codex_use_case import CodexRequestInput


class CodexIntegrationMixin:
    def _init_codex_state(self) -> None:
        self._codex_enabled: bool = False
        self._codex_proxy: str = "http://127.0.0.1:10808"
        self._codex_answer_keyword: str = "ANSWER"
        self._codex_timeout_s: int = 90
        self._codex_max_log_chars: int = 24000
        self._codex_command_tokens: List[str] = ["codex"]
        self._codex_path_hints: List[str] = []
        self._codex_profiles: List[CodexProfile] = []
        self._codex_selected_profile_id: str = ""
        self._codex_profile_buttons: Dict[str, QPushButton] = {}
        self._codex_busy: bool = False
        self._codex_ui_q: "queue.Queue[dict]" = queue.Queue(maxsize=120)
        self.codex_timer: Optional[QTimer] = None

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

        self.txt_codex = QTextEdit()
        self.txt_codex.setReadOnly(True)
        self.txt_codex.setPlaceholderText("Codex output will appear here")
        self.txt_codex.setLineWrapMode(QTextEdit.WidgetWidth)
        self.txt_codex.document().setMaximumBlockCount(1200)
        self.txt_codex.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        codex_layout.addWidget(self.txt_codex, 1)

        codex_input_row = QHBoxLayout()
        self.txt_codex_input = QLineEdit()
        self.txt_codex_input.setPlaceholderText("Type request or ANSWER")
        codex_input_row.addWidget(self.txt_codex_input, 1)
        self.btn_codex_send = QPushButton("Send")
        codex_input_row.addWidget(self.btn_codex_send, 0)
        codex_layout.addLayout(codex_input_row)

        self.grp_codex.setVisible(False)
        splitter.addWidget(self.grp_codex)
        splitter.setSizes([820, 0])

    def _connect_codex_signals(self) -> None:
        self.btn_codex_toggle.clicked.connect(self._toggle_codex_console)
        self.btn_codex_send.clicked.connect(self._on_codex_send_clicked)
        self.txt_codex_input.returnPressed.connect(self._on_codex_send_clicked)

    def _set_codex_inputs_enabled(self, enabled: bool) -> None:
        self.txt_codex_input.setEnabled(bool(enabled))
        self.btn_codex_send.setEnabled(bool(enabled))

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

    def _build_codex_config(self) -> Dict[str, Any]:
        return codex_settings_to_dict(self._codex_settings_from_ui())

    def _codex_settings_from_ui(self) -> CodexSettings:
        return CodexSettings(
            enabled=bool(self._codex_enabled),
            proxy=str(self._codex_proxy),
            answer_keyword=str(self._codex_answer_keyword),
            timeout_s=int(self._codex_timeout_s),
            max_log_chars=int(self._codex_max_log_chars),
            command_tokens=list(self._codex_command_tokens),
            path_hints=list(self._codex_path_hints),
            console_expanded=bool(self.btn_codex_toggle.isChecked()),
            selected_profile_id=str(self._codex_selected_profile_id or ""),
            profiles=list(self._codex_profiles),
        )

    def _apply_codex_settings(self, settings: CodexSettings) -> None:
        self._codex_proxy = str(settings.proxy)
        self._codex_answer_keyword = str(settings.answer_keyword or "ANSWER")
        self._codex_timeout_s = int(settings.timeout_s)
        self._codex_max_log_chars = int(settings.max_log_chars)
        self._codex_command_tokens = list(settings.command_tokens)
        self._codex_path_hints = list(settings.path_hints)
        self._codex_profiles = list(settings.profiles)

        self._refresh_codex_profile_buttons()
        self._set_codex_selected_profile(settings.selected_profile_id, mark_dirty=False)
        self._set_codex_enabled_ui(settings.enabled)
        self.btn_codex_toggle.setChecked(bool(settings.console_expanded))
        self._apply_codex_console_visibility(expanded=bool(settings.console_expanded))

    def _set_codex_enabled_ui(self, enabled: bool) -> None:
        self._codex_enabled = bool(enabled)
        self.btn_codex_toggle.setVisible(self._codex_enabled)
        if not self._codex_enabled:
            self.btn_codex_toggle.setChecked(False)
            self._apply_codex_console_visibility(expanded=False)
            self._set_codex_inputs_enabled(False)
            self.lbl_codex_status.setText("disabled by config")
            return
        self.lbl_codex_status.setText("idle")
        has_profiles = len(self._codex_profiles) > 0
        can_send = has_profiles and (not self._codex_busy)
        self._set_codex_inputs_enabled(can_send)

    def _clear_codex_profile_buttons(self) -> None:
        while self.codex_profiles_row.count() > 0:
            item = self.codex_profiles_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._codex_profile_buttons.clear()

    def _refresh_codex_profile_buttons(self) -> None:
        self._clear_codex_profile_buttons()
        if not self._codex_profiles:
            self.codex_profiles_row.addWidget(QLabel("No codex profiles in config"), 0)
            self.codex_profiles_row.addStretch(1)
            return

        for p in self._codex_profiles:
            btn = QPushButton(str(p.label))
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, pid=p.id: self._set_codex_selected_profile(pid, mark_dirty=checked))
            self.codex_profiles_row.addWidget(btn, 0)
            self._codex_profile_buttons[p.id] = btn
        self.codex_profiles_row.addStretch(1)

    def _set_codex_selected_profile(self, profile_id: str, *, mark_dirty: bool) -> None:
        pid = str(profile_id or "").strip()
        ids = {p.id for p in self._codex_profiles}
        if pid not in ids:
            pid = self._codex_profiles[0].id if self._codex_profiles else ""
        self._codex_selected_profile_id = pid

        for bid, btn in self._codex_profile_buttons.items():
            btn.blockSignals(True)
            btn.setChecked(bid == pid)
            btn.blockSignals(False)

        if mark_dirty:
            self._mark_config_dirty()

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

    def _load_codex_from_config(self, codex: Any) -> None:
        try:
            self._apply_codex_settings(parse_codex_settings(codex))
        except Exception:
            self._apply_codex_settings(CodexSettings(enabled=False))

    def _append_codex_line(self, line: str) -> None:
        max_chars = 180_000
        if self.txt_codex.document().characterCount() > max_chars:
            self.txt_codex.clear()
            self.txt_codex.append("[codex console cleared: too large]")

        self.txt_codex.append(str(line))
        self.txt_codex.moveCursor(QTextCursor.End)
        self.txt_codex.ensureCursorVisible()

    def _get_selected_codex_profile(self) -> Optional[CodexProfile]:
        for p in self._codex_profiles:
            if p.id == self._codex_selected_profile_id:
                return p
        return self._codex_profiles[0] if self._codex_profiles else None

    def _set_codex_busy(self, busy: bool) -> None:
        self._codex_busy = bool(busy)
        can_send = bool(self._codex_enabled and (not self._codex_busy) and len(self._codex_profiles) > 0)
        self._set_codex_inputs_enabled(can_send)
        if self._codex_enabled:
            self.lbl_codex_status.setText("busy..." if self._codex_busy else "idle")

    def _on_codex_send_clicked(self) -> None:
        if not self._codex_enabled:
            return
        if self._codex_busy:
            self._append_codex_line(f"[{self._fmt_ts(time.time())}] busy: wait for current request")
            return

        req = (self.txt_codex_input.text() or "").strip()
        if not req:
            return

        profile = self._get_selected_codex_profile()
        if profile is None:
            self._append_codex_line(f"[{self._fmt_ts(time.time())}] no codex profile configured")
            return

        self._append_codex_line(f"[{self._fmt_ts(time.time())}] you ({profile.label}): {req}")
        self.txt_codex_input.clear()
        self._set_codex_busy(True)

        self.background_task_runner.start(
            name="codex-helper-worker",
            target=self._run_codex_exec_worker,
            args=(profile, req),
        )

    def _codex_push_event(self, ev: Dict[str, Any]) -> None:
        try:
            self._codex_ui_q.put_nowait(ev)
        except queue.Full:
            try:
                _ = self._codex_ui_q.get_nowait()
            except Exception:
                pass
            try:
                self._codex_ui_q.put_nowait(ev)
            except Exception:
                pass

    def _run_codex_exec_worker(self, profile: CodexProfile, original_cmd: str) -> None:
        use_case = getattr(self, "codex_request_use_case", None)
        if use_case is None:
            self._codex_push_event(
                {
                    "type": "codex_result",
                    "ok": False,
                    "profile": profile.label,
                    "cmd": original_cmd,
                    "text": "codex request use case is not configured",
                    "dt_s": 0.0,
                }
            )
            return

        result = use_case.execute(
            CodexRequestInput(
                user_text=original_cmd,
                profile=profile,
                project_root=self.project_root,
                human_log_path=Path(self._human_log_path) if self._human_log_path is not None else None,
                human_log_fh=self._human_log_fh,
                max_log_chars=int(self._codex_max_log_chars),
                answer_keyword=self._codex_answer_keyword,
                execution_settings=CodexExecutionSettings(
                    command_tokens=list(self._codex_command_tokens),
                    path_hints=list(self._codex_path_hints),
                    proxy=str(self._codex_proxy or ""),
                    timeout_s=int(self._codex_timeout_s),
                ),
            )
        )
        self._codex_push_event(
            {
                "type": "codex_result",
                "ok": bool(result.ok),
                "profile": result.profile,
                "cmd": result.cmd,
                "text": result.text,
                "dt_s": float(result.dt_s),
            }
        )

    def _drain_codex_ui_events(self, limit: int = 8) -> None:
        n = 0
        while n < limit:
            try:
                ev = self._codex_ui_q.get_nowait()
            except queue.Empty:
                break
            n += 1

            if str(ev.get("type", "")) != "codex_result":
                continue

            ok = bool(ev.get("ok", False))
            profile = str(ev.get("profile", ""))
            dt_s = float(ev.get("dt_s", 0.0))
            text = str(ev.get("text", "")).strip()
            tss = self._fmt_ts(time.time())

            if ok:
                self._append_codex_line(f"[{tss}] codex ({profile}, {dt_s:.1f}s):")
                self._append_codex_line(text)
            else:
                self._append_codex_line(f"[{tss}] codex error ({profile}, {dt_s:.1f}s): {text}")

            self._set_codex_busy(False)
