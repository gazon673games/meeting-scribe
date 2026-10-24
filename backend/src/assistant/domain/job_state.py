from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from shared.domain.errors import InvalidJobTransition


class InvalidAssistantJobTransition(InvalidJobTransition):
    pass


class AssistantJobState(str, Enum):
    IDLE = 'idle'
    RUNNING = 'running'
    FALLBACK = 'fallback'


_IDLE_STATES = (AssistantJobState.IDLE,)
_ACTIVE_STATES = (AssistantJobState.RUNNING, AssistantJobState.FALLBACK)


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
        self._transition(_IDLE_STATES, AssistantJobState.RUNNING)

    def begin_fallback(self) -> None:
        self._transition(
            _ACTIVE_STATES,
            AssistantJobState.FALLBACK,
        )

    def finish(self) -> None:
        self._transition(
            _ACTIVE_STATES,
            AssistantJobState.IDLE,
        )

    def _transition(
        self, allowed: tuple[AssistantJobState, ...], target: AssistantJobState
    ) -> None:
        if self.state not in allowed:
            expected = ', '.join(sorted(state.value for state in allowed))
            raise InvalidAssistantJobTransition(
                f'Cannot transition assistant job from {self.state.value} to {target.value}; '
                f'expected one of: {expected}'
            )
        self.state = target
