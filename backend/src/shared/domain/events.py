from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DomainEvent:
    occurred_at: float = field(default_factory=time.time)


class AggregateRoot:
    def __init__(self) -> None:
        self._domain_events: list[DomainEvent] = []

    @property
    def domain_events(self) -> tuple[DomainEvent, ...]:
        return tuple(self._domain_events)

    def pull_domain_events(self) -> list[DomainEvent]:
        events = list(self._domain_events)
        self._domain_events.clear()
        return events

    def _record_domain_event(self, event: DomainEvent) -> None:
        self._domain_events.append(event)
