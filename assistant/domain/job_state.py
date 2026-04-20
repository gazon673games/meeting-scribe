from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from shared.domain.errors import InvalidJobTransition


class InvalidAssistantJobTransition(InvalidJobTransition):
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
            raise InvalidAssistantJobTransition(
                f"Cannot transition assistant job from {self.state.value} to {target.value}; "
                f"expected one of: {expected}"
            )
        self.state = target
