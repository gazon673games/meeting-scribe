from __future__ import annotations

from dataclasses import dataclass

from shared.domain.events import DomainEvent


@dataclass(frozen=True)
class TranscriptionStartRequested(DomainEvent):
    model_name: str = ""
    mode: str = ""
    language: str = ""


@dataclass(frozen=True)
class TranscriptionFallbackStarted(DomainEvent):
    attempt_label: str = ""
    model_name: str = ""
    reason: str = ""


@dataclass(frozen=True)
class TranscriptionStarted(DomainEvent):
    attempt_label: str = ""
    degraded: bool = False


@dataclass(frozen=True)
class TranscriptionStartFailed(DomainEvent):
    reason: str = ""


@dataclass(frozen=True)
class TranscriptionStopRequested(DomainEvent):
    pass


@dataclass(frozen=True)
class TranscriptionStopped(DomainEvent):
    stop_error: str = ""
