from __future__ import annotations

from dataclasses import dataclass

from shared.domain.events import DomainEvent


@dataclass(frozen=True)
class AssistantRequestStarted(DomainEvent):
    profile: str = ""
    source_label: str = ""


@dataclass(frozen=True)
class AssistantFallbackStarted(DomainEvent):
    profile: str = ""
    reason: str = ""


@dataclass(frozen=True)
class AssistantRequestFinished(DomainEvent):
    profile: str = ""
    ok: bool = False
    elapsed_s: float = 0.0
