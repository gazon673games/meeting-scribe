from __future__ import annotations

from typing import Any, Optional

from application.command_bus import CommandDispatcher
from application.commands import StartSessionCommand, StopSessionCommand, SwitchProfileCommand
from session.domain.aggregate import SessionAggregate
from transcription.domain.aggregate import TranscriptionJobAggregate
from ui.session_start_mixin import SessionStartMixin
from ui.session_stop_mixin import SessionStopMixin
from ui.session_ui_mixin import SessionUiMixin


class SessionMixin(SessionStartMixin, SessionStopMixin, SessionUiMixin):
    def _init_session_state(self) -> None:
        self.asr: Any = None
        self.asr_running: bool = False
        self._asr_overload_active: bool = False
        self._last_warn_ts: float = 0.0

        self._tap_dropped_total: int = 0
        self._seg_dropped_total: int = 0
        self._seg_skipped_total: int = 0
        self._avg_latency_s: float = 0.0
        self._p95_latency_s: float = 0.0
        self._lag_s: float = 0.0

        self._silence_eps: float = 1e-4
        self._silence_alert_s: float = 15.0
        self._desktop_silence_since_mono: Optional[float] = None

        self._long_run_mode: bool = False
        self._ui_interval_normal_ms: int = 120
        self._ui_interval_long_ms: int = 260

        self._session_state = SessionAggregate()
        self._transcription_state = TranscriptionJobAggregate()
        self._asr_supervision_report = None
        self._closing: bool = False
        self._offline_thread: Any = None
        self._asr_stop_thread: Any = None
        self._command_dispatcher = CommandDispatcher()
        self._command_dispatcher.register(StartSessionCommand, self._handle_start_session_command)
        self._command_dispatcher.register(StopSessionCommand, self._handle_stop_session_command)
        self._command_dispatcher.register(SwitchProfileCommand, self._handle_switch_profile_command)
