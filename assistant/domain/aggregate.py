from __future__ import annotations

from dataclasses import dataclass, field

from assistant.domain.events import AssistantFallbackStarted, AssistantRequestFinished, AssistantRequestStarted
from assistant.domain.job_state import AssistantJobState, AssistantJobStateMachine
from shared.domain.events import AggregateRoot


@dataclass(init=False)
class AssistantJobAggregate(AggregateRoot):
    state_machine: AssistantJobStateMachine = field(default_factory=AssistantJobStateMachine)

    def __init__(self, state_machine: AssistantJobStateMachine | None = None) -> None:
        super().__init__()
        self.state_machine = state_machine or AssistantJobStateMachine()

    @property
    def state(self) -> AssistantJobState:
        return self.state_machine.state

    @property
    def is_busy(self) -> bool:
        return self.state_machine.is_busy

    @property
    def is_fallback(self) -> bool:
        return self.state_machine.is_fallback

    def begin(self, *, profile: str = "", source_label: str = "") -> None:
        self.state_machine.begin()
        self._record_domain_event(AssistantRequestStarted(profile=str(profile), source_label=str(source_label)))

    def begin_fallback(self, *, profile: str = "", reason: str = "") -> None:
        self.state_machine.begin_fallback()
        self._record_domain_event(AssistantFallbackStarted(profile=str(profile), reason=str(reason)))

    def finish(self, *, profile: str = "", ok: bool = False, elapsed_s: float = 0.0) -> None:
        self.state_machine.finish()
        self._record_domain_event(
            AssistantRequestFinished(profile=str(profile), ok=bool(ok), elapsed_s=float(elapsed_s))
        )
