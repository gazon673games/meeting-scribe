from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InvalidJobTransition(RuntimeError):
    pass


class AssistantJobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FALLBACK = "fallback"


@dataclass
class AssistantJobStateMachine:
    state: AssistantJobState = AssistantJobState.IDLE

    @property
    def is_busy(self) -> bool:
        return self.state != AssistantJobState.IDLE

    @property
    def is_fallback(self) -> bool:
        return self.state == AssistantJobState.FALLBACK

    def begin(self) -> None:
        self._transition({AssistantJobState.IDLE}, AssistantJobState.RUNNING)

    def begin_fallback(self) -> None:
        self._transition({AssistantJobState.RUNNING, AssistantJobState.FALLBACK}, AssistantJobState.FALLBACK)

    def finish(self) -> None:
        self._transition({AssistantJobState.RUNNING, AssistantJobState.FALLBACK}, AssistantJobState.IDLE)

    def _transition(self, allowed: set[AssistantJobState], target: AssistantJobState) -> None:
        if self.state not in allowed:
            expected = ", ".join(sorted(state.value for state in allowed))
            raise InvalidJobTransition(
                f"Cannot transition assistant job from {self.state.value} to {target.value}; "
                f"expected one of: {expected}"
            )
        self.state = target


class TranscriptionJobState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    FALLBACK = "fallback"
    RUNNING = "running"
    RUNNING_DEGRADED = "running_degraded"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass
class TranscriptionJobStateMachine:
    state: TranscriptionJobState = TranscriptionJobState.IDLE

    @property
    def can_start(self) -> bool:
        return self.state in {TranscriptionJobState.IDLE, TranscriptionJobState.FAILED}

    @property
    def can_stop(self) -> bool:
        return self.state in {TranscriptionJobState.RUNNING, TranscriptionJobState.RUNNING_DEGRADED}

    @property
    def is_running(self) -> bool:
        return self.state in {TranscriptionJobState.RUNNING, TranscriptionJobState.RUNNING_DEGRADED}

    @property
    def is_degraded(self) -> bool:
        return self.state == TranscriptionJobState.RUNNING_DEGRADED

    @property
    def is_stopping(self) -> bool:
        return self.state == TranscriptionJobState.STOPPING

    def begin_start(self) -> None:
        self._transition({TranscriptionJobState.IDLE, TranscriptionJobState.FAILED}, TranscriptionJobState.STARTING)

    def begin_fallback(self) -> None:
        self._transition(
            {TranscriptionJobState.STARTING, TranscriptionJobState.FALLBACK},
            TranscriptionJobState.FALLBACK,
        )

    def finish_start(self, *, degraded: bool = False) -> None:
        target = TranscriptionJobState.RUNNING_DEGRADED if degraded else TranscriptionJobState.RUNNING
        self._transition({TranscriptionJobState.STARTING, TranscriptionJobState.FALLBACK}, target)

    def fail_start(self) -> None:
        self._transition({TranscriptionJobState.STARTING, TranscriptionJobState.FALLBACK}, TranscriptionJobState.FAILED)

    def begin_stop(self) -> None:
        self._transition(
            {TranscriptionJobState.RUNNING, TranscriptionJobState.RUNNING_DEGRADED},
            TranscriptionJobState.STOPPING,
        )

    def finish_stop(self) -> None:
        self._transition({TranscriptionJobState.STOPPING}, TranscriptionJobState.IDLE)

    def reset(self) -> None:
        self.state = TranscriptionJobState.IDLE

    def _transition(self, allowed: set[TranscriptionJobState], target: TranscriptionJobState) -> None:
        if self.state not in allowed:
            expected = ", ".join(sorted(state.value for state in allowed))
            raise InvalidJobTransition(
                f"Cannot transition transcription job from {self.state.value} to {target.value}; "
                f"expected one of: {expected}"
            )
        self.state = target
