from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QPushButton

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
from assistant.application.service import AssistantRuntimeOptions
from assistant.domain.aggregate import AssistantJobAggregate
from ui.codex_context_mixin import CODEX_CONTEXT_TRANSCRIPT, CodexContextMixin
from ui.codex_panel_mixin import CodexPanelMixin
from ui.codex_profile_state import CodexProfileState

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


class CodexIntegrationMixin(CodexPanelMixin, CodexContextMixin):
    def _init_codex_state(self) -> None:
        self._codex_enabled: bool = False
        self._codex_proxy: str = "http://127.0.0.1:10808"
        self._codex_answer_keyword: str = "ANSWER"
        self._codex_timeout_s: int = 90
        self._codex_max_log_chars: int = 24000
        self._codex_command_tokens: List[str] = ["codex"]
        self._codex_path_hints: List[str] = []
        self._codex_context_source: str = CODEX_CONTEXT_TRANSCRIPT
        self._profile_state = CodexProfileState()
        self._assistant_job_state = AssistantJobAggregate()
        self._assistant_supervision_report = None
        self._codex_busy: bool = False
        self._codex_event_bus = QueuedEventBus(maxsize=120)
        self._codex_event_bus.subscribe(CodexFallbackStartedEvent, self._handle_codex_fallback_event)
        self._codex_event_bus.subscribe(CodexResultEvent, self._handle_codex_result_event)
        self.codex_timer: Optional[QTimer] = None

        dispatcher = getattr(self, "_command_dispatcher", None)
        if dispatcher is not None:
            dispatcher.register(InvokeAssistantCommand, self._handle_invoke_assistant_command)

    # ── public shims used by config / panel mixins ─────────────────────

    @property
    def _codex_profiles(self) -> List[CodexProfile]:
        return self._profile_state.profiles

    @property
    def _codex_selected_profile_id(self) -> str:
        return self._profile_state.selected_id

    @property
    def _codex_profile_buttons(self) -> Dict[str, QPushButton]:
        return self._profile_state.buttons

    # ── config serialization ───────────────────────────────────────────

    def _build_codex_config(self) -> Dict[str, Any]:
        return codex_settings_to_dict(self._codex_settings_from_ui())

    def _codex_settings_from_ui(self) -> CodexSettings:
        return CodexSettings(
            enabled=self._codex_enabled,
            proxy=self._codex_proxy,
            answer_keyword=self._codex_answer_keyword,
            timeout_s=self._codex_timeout_s,
            max_log_chars=self._codex_max_log_chars,
            command_tokens=list(self._codex_command_tokens),
            path_hints=list(self._codex_path_hints),
            context_source=self._codex_context_source_from_ui(),
            console_expanded=bool(self.btn_codex_toggle.isChecked()),
            selected_profile_id=self._profile_state.selected_id,
            profiles=list(self._profile_state.profiles),
        )

    def _apply_codex_settings(self, settings: CodexSettings) -> None:
        self._codex_proxy = str(settings.proxy)
        self._codex_answer_keyword = str(settings.answer_keyword or "ANSWER")
        self._codex_timeout_s = int(settings.timeout_s)
        self._codex_max_log_chars = int(settings.max_log_chars)
        self._codex_command_tokens = list(settings.command_tokens)
        self._codex_path_hints = list(settings.path_hints)
        self._set_codex_context_source(settings.context_source)
        self._profile_state.set_profiles(settings.profiles)

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
        self._set_codex_inputs_enabled(bool(self._profile_state.profiles and not self._codex_busy))

    def _load_codex_from_config(self, codex: Any) -> None:
        try:
            self._apply_codex_settings(parse_codex_settings(codex))
        except Exception:
            self._apply_codex_settings(CodexSettings(enabled=False))

    # ── profile buttons ────────────────────────────────────────────────

    def _clear_codex_profile_buttons(self) -> None:
        while self.codex_profiles_row.count() > 0:
            item = self.codex_profiles_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._profile_state.buttons.clear()

    def _refresh_codex_profile_buttons(self) -> None:
        self._clear_codex_profile_buttons()
        if not self._profile_state.profiles:
            self.codex_profiles_row.addWidget(QLabel("No codex profiles in config"), 0)
            self.codex_profiles_row.addStretch(1)
            return
        for profile in self._profile_state.profiles:
            btn = QPushButton(str(profile.label))
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, pid=profile.id: self._set_codex_selected_profile(pid, mark_dirty=checked))
            self.codex_profiles_row.addWidget(btn, 0)
            self._profile_state.buttons[profile.id] = btn
        self.codex_profiles_row.addStretch(1)

    def _set_codex_selected_profile(self, profile_id: str, *, mark_dirty: bool) -> None:
        self._profile_state.select(profile_id)
        self._profile_state.sync_buttons()
        if mark_dirty:
            self._mark_config_dirty()

    # ── dispatch helper ────────────────────────────────────────────────

    def _dispatch(self, command) -> None:
        dispatcher = getattr(self, "_command_dispatcher", None)
        if dispatcher is not None:
            dispatcher.dispatch(command)
        else:
            self._handle_invoke_assistant_command(command)

    # ── request entry points ───────────────────────────────────────────

    def _on_codex_send_clicked(self) -> None:
        req = (self.txt_codex_input.text() or "").strip()
        if not req:
            return
        self._start_codex_request(profile=self._profile_state.selected_profile(), request_text=req, source_label="you")
        self.txt_codex_input.clear()

    def _on_codex_answer_clicked(self) -> None:
        self._start_codex_request(
            profile=self._profile_state.policy_profile(PROFILE_REALTIME),
            request_text=self._codex_answer_keyword,
            source_label="answer",
            max_log_chars=FAST_ANSWER_LOG_CHARS,
            timeout_s=FAST_ANSWER_TIMEOUT_S,
            fallback_max_log_chars=FAST_ANSWER_FALLBACK_LOG_CHARS,
            fallback_timeout_s=FAST_ANSWER_FALLBACK_TIMEOUT_S,
        )

    def _on_codex_summary_clicked(self) -> None:
        self._start_codex_request(
            profile=self._profile_state.policy_profile(PROFILE_QUALITY),
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

        context = self._resolve_codex_context()
        command = InvokeAssistantCommand(
            profile=profile,
            request_text=req,
            source_label=source_label,
            context_source=context.source,
            context_label=context.label,
            context_text=context.text,
            human_log_path=context.human_log_path,
            human_log_fh=context.human_log_fh,
            max_log_chars=max_log_chars,
            timeout_s=timeout_s,
            fallback_max_log_chars=fallback_max_log_chars,
            fallback_timeout_s=fallback_timeout_s,
        )
        self._dispatch(command)

    def _handle_invoke_assistant_command(self, command: InvokeAssistantCommand) -> None:
        self._append_codex_line(
            f"[{self._fmt_ts(time.time())}] {command.source_label} "
            f"({command.profile.label}, {command.context_label}): {command.request_text}"
        )
        self._set_codex_busy(True)
        self.background_task_runner.start(name="codex-helper-worker", target=self._run_codex_exec_worker, args=(command,))

    # ── worker ─────────────────────────────────────────────────────────

    def _codex_push_event(self, ev: object) -> None:
        self._codex_event_bus.publish(ev)

    def _run_codex_exec_worker(self, command: InvokeAssistantCommand) -> None:
        service = getattr(self, "assistant_service", None)
        if service is None:
            self._codex_push_event(CodexResultEvent(
                ok=False, profile=command.profile.label,
                cmd=command.request_text, text="assistant service is not configured", dt_s=0.0,
            ))
            return
        service.execute(command, options=self._assistant_runtime_options(), publish_event=self._codex_push_event)
        # Store under a lock-free write; UI thread reads this only after event arrives.
        self._assistant_supervision_report = service.last_supervision_report

    def _assistant_runtime_options(self) -> AssistantRuntimeOptions:
        return AssistantRuntimeOptions(
            project_root=self.project_root,
            default_max_log_chars=int(self._codex_max_log_chars),
            answer_keyword=str(self._codex_answer_keyword),
            command_tokens=list(self._codex_command_tokens),
            path_hints=list(self._codex_path_hints),
            proxy=str(self._codex_proxy or ""),
            default_timeout_s=int(self._codex_timeout_s),
        )

    # ── busy state ─────────────────────────────────────────────────────

    def _set_codex_busy(self, busy: bool) -> None:
        if busy and not self._assistant_job_state.is_busy:
            self._assistant_job_state.begin(profile=str(self._profile_state.selected_id), source_label="ui")
        elif not busy and self._assistant_job_state.is_busy:
            self._assistant_job_state.finish()
        self._codex_busy = self._assistant_job_state.is_busy
        can_send = bool(self._codex_enabled and not self._codex_busy and self._profile_state.profiles)
        self._set_codex_inputs_enabled(can_send)
        if self._codex_enabled:
            self.lbl_codex_status.setText("busy..." if self._codex_busy else "idle")

    def _set_codex_fallback_active(self) -> None:
        self._assistant_job_state.begin_fallback(reason="codex fallback")
        self._codex_busy = True
        self._set_codex_inputs_enabled(False)
        if self._codex_enabled:
            self.lbl_codex_status.setText("fallback...")

    # ── event handlers ─────────────────────────────────────────────────

    def _drain_codex_ui_events(self, limit: int = 8) -> None:
        self._codex_event_bus.drain(limit=int(limit))

    def _handle_codex_fallback_event(self, ev: CodexFallbackStartedEvent) -> None:
        self._append_codex_line(f"[{self._fmt_ts(time.time())}] codex fallback: {str(ev.reason).strip()}")
        self._set_codex_fallback_active()

    def _handle_codex_result_event(self, ev: CodexResultEvent) -> None:
        tss = self._fmt_ts(time.time())
        profile = str(ev.profile)
        text = str(ev.text).strip()
        dt = f"{float(ev.dt_s):.1f}s"
        if ev.ok:
            self._append_codex_line(f"[{tss}] codex ({profile}, {dt}):")
            self._append_codex_line(text)
        else:
            self._append_codex_line(f"[{tss}] codex error ({profile}, {dt}): {text}")
        self._set_codex_busy(False)
