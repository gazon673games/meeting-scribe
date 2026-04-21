from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SessionState(str, Enum):
    IDLE = "idle"
    DOWNLOADING_MODEL = "downloading_model"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    OFFLINE_PASS = "offline_pass"


class InvalidSessionTransition(RuntimeError):
    pass


@dataclass
class SessionStateMachine:
    state: SessionState = SessionState.IDLE

    @property
    def can_start(self) -> bool:
        return self.state == SessionState.IDLE

    @property
    def can_stop(self) -> bool:
        return self.state == SessionState.RUNNING

    @property
    def is_running(self) -> bool:
        return self.state == SessionState.RUNNING

    @property
    def is_stopping(self) -> bool:
        return self.state == SessionState.STOPPING

    @property
    def is_offline_pass(self) -> bool:
        return self.state == SessionState.OFFLINE_PASS

    @property
    def is_downloading_model(self) -> bool:
        return self.state == SessionState.DOWNLOADING_MODEL

    def begin_model_download(self) -> None:
        self._transition({SessionState.IDLE}, SessionState.DOWNLOADING_MODEL)

    def finish_model_download(self) -> None:
        self._transition({SessionState.DOWNLOADING_MODEL}, SessionState.IDLE)

    def begin_start(self) -> None:
        self._transition({SessionState.IDLE}, SessionState.STARTING)

    def finish_start(self) -> None:
        self._transition({SessionState.STARTING}, SessionState.RUNNING)

    def fail_start(self) -> None:
        self._transition({SessionState.STARTING}, SessionState.IDLE)

    def begin_stop(self) -> None:
        self._transition({SessionState.RUNNING}, SessionState.STOPPING)

    def finish_stop(self) -> None:
        self._transition({SessionState.STOPPING}, SessionState.IDLE)

    def begin_offline_pass(self) -> None:
        self._transition({SessionState.IDLE}, SessionState.OFFLINE_PASS)

    def finish_offline_pass(self) -> None:
        self._transition({SessionState.OFFLINE_PASS}, SessionState.IDLE)

    def _transition(self, allowed: set[SessionState], target: SessionState) -> None:
        if self.state not in allowed:
            expected = ", ".join(sorted(state.value for state in allowed))
            raise InvalidSessionTransition(
                f"Cannot transition session from {self.state.value} to {target.value}; "
                f"expected one of: {expected}"
            )
        self.state = target
