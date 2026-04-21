from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from shared.domain.errors import InvalidJobTransition


class InvalidTranscriptionJobTransition(InvalidJobTransition):
    pass


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
            raise InvalidTranscriptionJobTransition(
                f"Cannot transition transcription job from {self.state.value} to {target.value}; "
                f"expected one of: {expected}"
            )
        self.state = target
