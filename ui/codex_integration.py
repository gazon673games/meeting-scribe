from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

from application.asr_profiles import PROFILE_QUALITY, PROFILE_REALTIME
from application.codex_config import (
    CodexProfile,
    CodexSettings,
    codex_settings_to_dict,
    parse_codex_settings,
)
from application.commands import InvokeAssistantCommand
from application.event_bus import QueuedEventBus
from application.event_types import CodexFallbackStartedEvent, CodexResultEvent
from application.model_policy import ModelOrchestrator
from assistant.application.service import AssistantRuntimeOptions
from assistant.domain.aggregate import AssistantJobAggregate

FAST_ANSWER_LOG_CHARS = 4000
FAST_ANSWER_TIMEOUT_S = 35
FAST_ANSWER_FALLBACK_LOG_CHARS = 2000
FAST_ANSWER_FALLBACK_TIMEOUT_S = 20
SUMMARY_LOG_CHARS = 200000
SUMMARY_TIMEOUT_S = 180
SUMMARY_REQUEST = (
    "Summarize the whole session context. "
    "Focus on decisions, open questions, risks, and the most useful next actions."
)
CODEX_CONTEXT_TRANSCRIPT = "transcript"
CODEX_CONTEXT_CURRENT_HUMAN_LOG = "current_human_log"
CODEX_CONTEXT_LATEST_HUMAN_LOG = "latest_human_log"
CODEX_CONTEXT_FILE_PREFIX = "human_log:"


class CodexIntegrationMixin:
    def _init_codex_state(self) -> None:
        self._codex_enabled: bool = False
        self._codex_proxy: str = "http://127.0.0.1:10808"
        self._codex_answer_keyword: str = "ANSWER"
        self._codex_timeout_s: int = 90
        self._codex_max_log_chars: int = 24000
        self._codex_command_tokens: List[str] = ["codex"]
        self._codex_path_hints: List[str] = []
        self._codex_context_source: str = CODEX_CONTEXT_TRANSCRIPT
        self._codex_profiles: List[CodexProfile] = []
        self._codex_selected_profile_id: str = ""
        self._codex_profile_buttons: Dict[str, QPushButton] = {}
        self._assistant_job_state = AssistantJobAggregate()
        self._assistant_supervision_report = None
        self._codex_busy: bool = False
        self._codex_event_bus = QueuedEventBus(maxsize=120)
        self._codex_event_bus.subscribe(CodexFallbackStartedEvent, self._handle_codex_fallback_event)
        self._codex_event_bus.subscribe(CodexResultEvent, self._handle_codex_result_event)
        dispatcher = getattr(self, "_command_dispatcher", None)
        if dispatcher is not None:
            dispatcher.register(InvokeAssistantCommand, self._handle_invoke_assistant_command)
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

        context_row = QHBoxLayout()
        context_row.addWidget(QLabel("Context:"))
        self.cmb_codex_context = QComboBox()
        context_row.addWidget(self.cmb_codex_context, 1)
        self.btn_codex_context_refresh = QPushButton("Refresh")
        context_row.addWidget(self.btn_codex_context_refresh, 0)
        codex_layout.addLayout(context_row)
        self._refresh_codex_context_sources()

        codex_actions_row = QHBoxLayout()
        self.btn_codex_answer = QPushButton("Answer")
        self.btn_codex_answer.setToolTip("Fast answer from recent context")
        codex_actions_row.addWidget(self.btn_codex_answer, 0)

        self.btn_codex_summary = QPushButton("Summary")
        self.btn_codex_summary.setToolTip("Deep summary from full context")
        codex_actions_row.addWidget(self.btn_codex_summary, 0)
        codex_actions_row.addStretch(1)
        codex_layout.addLayout(codex_actions_row)

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
            context_source=self._codex_context_source_from_ui(),
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
        self._set_codex_context_source(settings.context_source)
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

    def _refresh_codex_context_sources(self) -> None:
        selected = self._codex_context_source_from_ui(default=self._codex_context_source)
        self.cmb_codex_context.blockSignals(True)
        self.cmb_codex_context.clear()
        self.cmb_codex_context.addItem("Current transcript", CODEX_CONTEXT_TRANSCRIPT)
        self.cmb_codex_context.addItem("Current session human log", CODEX_CONTEXT_CURRENT_HUMAN_LOG)
        self.cmb_codex_context.addItem("Latest human log", CODEX_CONTEXT_LATEST_HUMAN_LOG)

        logs_dir = self.project_root / "human_logs"
        try:
            files = [x for x in logs_dir.glob("chat_*.txt") if x.is_file()]
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            files = []

        for path in files[:30]:
            self.cmb_codex_context.addItem(f"Log: {path.name}", f"{CODEX_CONTEXT_FILE_PREFIX}{path.name}")

        self._set_codex_context_source(selected, mark_dirty=False)
        self.cmb_codex_context.blockSignals(False)

    def _codex_context_source_from_ui(self, default: str = CODEX_CONTEXT_TRANSCRIPT) -> str:
        combo = getattr(self, "cmb_codex_context", None)
        if combo is None:
            return str(default or CODEX_CONTEXT_TRANSCRIPT)
        data = combo.currentData()
        return str(data or default or CODEX_CONTEXT_TRANSCRIPT)

    def _set_codex_context_source(self, source: str, *, mark_dirty: bool = False) -> None:
        wanted = str(source or CODEX_CONTEXT_TRANSCRIPT)
        idx = self.cmb_codex_context.findData(wanted)
        if idx < 0 and wanted.startswith(CODEX_CONTEXT_FILE_PREFIX):
            name = Path(wanted[len(CODEX_CONTEXT_FILE_PREFIX):]).name
            path = self.project_root / "human_logs" / name
            if path.exists():
                self.cmb_codex_context.addItem(f"Log: {path.name}", f"{CODEX_CONTEXT_FILE_PREFIX}{path.name}")
                idx = self.cmb_codex_context.findData(f"{CODEX_CONTEXT_FILE_PREFIX}{path.name}")
            else:
                idx = self.cmb_codex_context.findData(CODEX_CONTEXT_LATEST_HUMAN_LOG)
        if idx < 0:
            idx = self.cmb_codex_context.findData(CODEX_CONTEXT_TRANSCRIPT)
        if idx >= 0:
            self.cmb_codex_context.setCurrentIndex(idx)
        self._codex_context_source = self._codex_context_source_from_ui(default=wanted)
        if mark_dirty:
            self._mark_config_dirty()

    def _on_codex_context_changed(self) -> None:
        self._codex_context_source = self._codex_context_source_from_ui()
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

    def _get_codex_policy_profile(self, policy_profile: str) -> Optional[CodexProfile]:
        profile_id = ModelOrchestrator().recommend_codex_profile_id(
            asr_profile=policy_profile,
            profiles=list(self._codex_profiles),
            current_profile_id=str(self._codex_selected_profile_id or ""),
        )
        for profile in self._codex_profiles:
            if profile.id == profile_id:
                return profile
        return self._get_selected_codex_profile()

    def _set_codex_busy(self, busy: bool) -> None:
        if busy and not self._assistant_job_state.is_busy:
            profile = getattr(self, "_codex_selected_profile_id", "")
            self._assistant_job_state.begin(profile=str(profile), source_label="ui")
        elif not busy and self._assistant_job_state.is_busy:
            self._assistant_job_state.finish()
        self._codex_busy = self._assistant_job_state.is_busy
        can_send = bool(self._codex_enabled and (not self._codex_busy) and len(self._codex_profiles) > 0)
        self._set_codex_inputs_enabled(can_send)
        if self._codex_enabled:
            self.lbl_codex_status.setText("busy..." if self._codex_busy else "idle")

    def _set_codex_fallback_active(self) -> None:
        self._assistant_job_state.begin_fallback(reason="codex fallback")
        self._codex_busy = True
        self._set_codex_inputs_enabled(False)
        if self._codex_enabled:
            self.lbl_codex_status.setText("fallback...")

    def _on_codex_send_clicked(self) -> None:
        req = (self.txt_codex_input.text() or "").strip()
        if not req:
            return

        profile = self._get_selected_codex_profile()
        self._start_codex_request(profile=profile, request_text=req, source_label="you")
        self.txt_codex_input.clear()

    def _on_codex_answer_clicked(self) -> None:
        profile = self._get_codex_policy_profile(PROFILE_REALTIME)
        self._start_codex_request(
            profile=profile,
            request_text=self._codex_answer_keyword,
            source_label="answer",
            max_log_chars=FAST_ANSWER_LOG_CHARS,
            timeout_s=FAST_ANSWER_TIMEOUT_S,
            fallback_max_log_chars=FAST_ANSWER_FALLBACK_LOG_CHARS,
            fallback_timeout_s=FAST_ANSWER_FALLBACK_TIMEOUT_S,
        )

    def _on_codex_summary_clicked(self) -> None:
        profile = self._get_codex_policy_profile(PROFILE_QUALITY)
        self._start_codex_request(
            profile=profile,
            request_text=SUMMARY_REQUEST,
            source_label="summary",
            max_log_chars=SUMMARY_LOG_CHARS,
            timeout_s=SUMMARY_TIMEOUT_S,
        )

    def _start_codex_request(
        self,
        *,
        profile: Optional[CodexProfile],
        request_text: str,
        source_label: str,
        max_log_chars: Optional[int] = None,
        timeout_s: Optional[int] = None,
        fallback_max_log_chars: Optional[int] = None,
        fallback_timeout_s: Optional[int] = None,
    ) -> None:
        if not self._codex_enabled:
            return
        if self._codex_busy:
            self._append_codex_line(f"[{self._fmt_ts(time.time())}] busy: wait for current request")
            return

        req = str(request_text or "").strip()
        if not req:
            return

        if profile is None:
            self._append_codex_line(f"[{self._fmt_ts(time.time())}] no codex profile configured")
            return

        context_text, human_log_path, human_log_fh, context_label = self._resolve_codex_context()
        command = InvokeAssistantCommand(
            profile=profile,
            request_text=req,
            source_label=source_label,
            context_source=self._codex_context_source_from_ui(),
            context_label=context_label,
            context_text=context_text,
            human_log_path=human_log_path,
            human_log_fh=human_log_fh,
            max_log_chars=max_log_chars,
            timeout_s=timeout_s,
            fallback_max_log_chars=fallback_max_log_chars,
            fallback_timeout_s=fallback_timeout_s,
        )

        dispatcher = getattr(self, "_command_dispatcher", None)
        if dispatcher is None:
            self._handle_invoke_assistant_command(command)
            return
        dispatcher.dispatch(command)

    def _handle_invoke_assistant_command(self, command: InvokeAssistantCommand) -> None:
        self._append_codex_line(
            f"[{self._fmt_ts(time.time())}] {command.source_label} "
            f"({command.profile.label}, {command.context_label}): {command.request_text}"
        )
        self._set_codex_busy(True)

        self.background_task_runner.start(
            name="codex-helper-worker",
            target=self._run_codex_exec_worker,
            args=(command,),
        )

    def _resolve_codex_context(self) -> tuple[Optional[str], Optional[Path], Any, str]:
        source = self._codex_context_source_from_ui()
        self._codex_context_source = source

        if source == CODEX_CONTEXT_TRANSCRIPT:
            return self.txt_tr.toPlainText(), None, None, "current transcript"

        if source == CODEX_CONTEXT_CURRENT_HUMAN_LOG:
            self._sync_transcript_store_refs()
            if self._human_log_path is not None:
                try:
                    if self._human_log_fh is not None:
                        self._human_log_fh.flush()
                except Exception:
                    pass
                return None, Path(self._human_log_path), self._human_log_fh, "current human log"
            return "", None, None, "current human log (empty)"

        if source.startswith(CODEX_CONTEXT_FILE_PREFIX):
            name = Path(source[len(CODEX_CONTEXT_FILE_PREFIX):]).name
            path = self.project_root / "human_logs" / name
            if not path.exists():
                return "", None, None, f"human log {name} (missing)"
            return None, path, None, f"human log {name}"

        return None, None, None, "latest human log"

    def _codex_push_event(self, ev: object) -> None:
        self._codex_event_bus.publish(ev)

    def _run_codex_exec_worker(self, command: InvokeAssistantCommand) -> None:
        service = getattr(self, "assistant_service", None)
        if service is None:
            self._codex_push_event(
                CodexResultEvent(
                    ok=False,
                    profile=command.profile.label,
                    cmd=command.request_text,
                    text="assistant service is not configured",
                    dt_s=0.0,
                )
            )
            return

        service.execute(
            command,
            options=AssistantRuntimeOptions(
                project_root=self.project_root,
                default_max_log_chars=int(self._codex_max_log_chars),
                answer_keyword=str(self._codex_answer_keyword),
                command_tokens=list(self._codex_command_tokens),
                path_hints=list(self._codex_path_hints),
                proxy=str(self._codex_proxy or ""),
                default_timeout_s=int(self._codex_timeout_s),
            ),
            publish_event=self._codex_push_event,
        )
        self._assistant_supervision_report = service.last_supervision_report

    def _drain_codex_ui_events(self, limit: int = 8) -> None:
        self._codex_event_bus.drain(limit=int(limit))

    def _handle_codex_fallback_event(self, ev: CodexFallbackStartedEvent) -> None:
        reason = str(ev.reason).strip()
        self._append_codex_line(f"[{self._fmt_ts(time.time())}] codex fallback: {reason}")
        self._set_codex_fallback_active()

    def _handle_codex_result_event(self, ev: CodexResultEvent) -> None:
        ok = bool(ev.ok)
        profile = str(ev.profile)
        dt_s = float(ev.dt_s)
        text = str(ev.text).strip()
        tss = self._fmt_ts(time.time())

        if ok:
            self._append_codex_line(f"[{tss}] codex ({profile}, {dt_s:.1f}s):")
            self._append_codex_line(text)
        else:
            self._append_codex_line(f"[{tss}] codex error ({profile}, {dt_s:.1f}s): {text}")

        self._set_codex_busy(False)
