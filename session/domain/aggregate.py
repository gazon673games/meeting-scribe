from __future__ import annotations

from dataclasses import dataclass, field

from shared.domain.events import AggregateRoot
from session.domain.events import (
    OfflinePassFinished,
    OfflinePassStarted,
    SessionModelDownloadFinished,
    SessionModelDownloadStarted,
    SessionStartFailed,
    SessionStartRequested,
    SessionStarted,
    SessionStopRequested,
    SessionStopped,
)
from session.domain.state import SessionState, SessionStateMachine


@dataclass(init=False)
class SessionAggregate(AggregateRoot):
    state_machine: SessionStateMachine = field(default_factory=SessionStateMachine)

    def __init__(self, state_machine: SessionStateMachine | None = None) -> None:
        super().__init__()
        self.state_machine = state_machine or SessionStateMachine()

    @property
    def state(self) -> SessionState:
        return self.state_machine.state

    @property
    def can_start(self) -> bool:
        return self.state_machine.can_start

    @property
    def can_stop(self) -> bool:
        return self.state_machine.can_stop

    @property
    def is_running(self) -> bool:
        return self.state_machine.is_running

    @property
    def is_stopping(self) -> bool:
        return self.state_machine.is_stopping

    @property
    def is_offline_pass(self) -> bool:
        return self.state_machine.is_offline_pass

    @property
    def is_downloading_model(self) -> bool:
        return self.state_machine.is_downloading_model

    def begin_model_download(self, model_name: str = "") -> None:
        self.state_machine.begin_model_download()
        self._record_domain_event(SessionModelDownloadStarted(model_name=str(model_name)))

    def finish_model_download(self, model_name: str = "", error: str = "") -> None:
        self.state_machine.finish_model_download()
        self._record_domain_event(SessionModelDownloadFinished(model_name=str(model_name), error=str(error)))

    def begin_start(
        self,
        *,
        source_count: int = 0,
        asr_enabled: bool = False,
        profile: str = "",
        language: str = "",
    ) -> None:
        self.state_machine.begin_start()
        self._record_domain_event(
            SessionStartRequested(
                source_count=int(source_count),
                asr_enabled=bool(asr_enabled),
                profile=str(profile),
                language=str(language),
            )
        )

    def finish_start(self, *, asr_running: bool = False, wav_recording: bool = False) -> None:
        self.state_machine.finish_start()
        self._record_domain_event(SessionStarted(asr_running=bool(asr_running), wav_recording=bool(wav_recording)))

    def fail_start(self, reason: str = "") -> None:
        self.state_machine.fail_start()
        self._record_domain_event(SessionStartFailed(reason=str(reason)))

    def begin_stop(self, *, run_offline_pass: bool = True) -> None:
        self.state_machine.begin_stop()
        self._record_domain_event(SessionStopRequested(run_offline_pass=bool(run_offline_pass)))

    def finish_stop(self, stop_error: str = "") -> None:
        self.state_machine.finish_stop()
        self._record_domain_event(SessionStopped(stop_error=str(stop_error or "")))

    def begin_offline_pass(self, wav_path: str = "") -> None:
        self.state_machine.begin_offline_pass()
        self._record_domain_event(OfflinePassStarted(wav_path=str(wav_path)))

    def finish_offline_pass(self, *, out_txt: str = "", error: str = "") -> None:
        self.state_machine.finish_offline_pass()
        self._record_domain_event(OfflinePassFinished(out_txt=str(out_txt), error=str(error)))
