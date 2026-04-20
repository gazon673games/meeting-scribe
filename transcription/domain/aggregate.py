from __future__ import annotations

from dataclasses import dataclass, field

from shared.domain.events import AggregateRoot
from transcription.domain.events import (
    TranscriptionFallbackStarted,
    TranscriptionStartFailed,
    TranscriptionStartRequested,
    TranscriptionStarted,
    TranscriptionStopRequested,
    TranscriptionStopped,
)
from transcription.domain.job_state import TranscriptionJobState, TranscriptionJobStateMachine


@dataclass(init=False)
class TranscriptionJobAggregate(AggregateRoot):
    state_machine: TranscriptionJobStateMachine = field(default_factory=TranscriptionJobStateMachine)

    def __init__(self, state_machine: TranscriptionJobStateMachine | None = None) -> None:
        super().__init__()
        self.state_machine = state_machine or TranscriptionJobStateMachine()

    @property
    def state(self) -> TranscriptionJobState:
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
    def is_degraded(self) -> bool:
        return self.state_machine.is_degraded

    @property
    def is_stopping(self) -> bool:
        return self.state_machine.is_stopping

    def begin_start(self, *, model_name: str = "", mode: str = "", language: str = "") -> None:
        self.state_machine.begin_start()
        self._record_domain_event(
            TranscriptionStartRequested(model_name=str(model_name), mode=str(mode), language=str(language))
        )

    def begin_fallback(self, *, attempt_label: str = "", model_name: str = "", reason: str = "") -> None:
        self.state_machine.begin_fallback()
        self._record_domain_event(
            TranscriptionFallbackStarted(
                attempt_label=str(attempt_label),
                model_name=str(model_name),
                reason=str(reason),
            )
        )

    def finish_start(self, *, degraded: bool = False, attempt_label: str = "") -> None:
        self.state_machine.finish_start(degraded=degraded)
        self._record_domain_event(TranscriptionStarted(attempt_label=str(attempt_label), degraded=bool(degraded)))

    def fail_start(self, reason: str = "") -> None:
        self.state_machine.fail_start()
        self._record_domain_event(TranscriptionStartFailed(reason=str(reason)))

    def begin_stop(self) -> None:
        self.state_machine.begin_stop()
        self._record_domain_event(TranscriptionStopRequested())

    def finish_stop(self, stop_error: str = "") -> None:
        self.state_machine.finish_stop()
        self._record_domain_event(TranscriptionStopped(stop_error=str(stop_error or "")))

    def reset(self) -> None:
        self.state_machine.reset()
