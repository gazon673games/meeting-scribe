from __future__ import annotations

from dataclasses import dataclass

from shared.domain.events import DomainEvent


@dataclass(frozen=True)
class SessionModelDownloadStarted(DomainEvent):
    model_name: str = ""


@dataclass(frozen=True)
class SessionModelDownloadFinished(DomainEvent):
    model_name: str = ""
    error: str = ""


@dataclass(frozen=True)
class SessionStartRequested(DomainEvent):
    source_count: int = 0
    asr_enabled: bool = False
    profile: str = ""
    language: str = ""


@dataclass(frozen=True)
class SessionStarted(DomainEvent):
    asr_running: bool = False
    wav_recording: bool = False


@dataclass(frozen=True)
class SessionStartFailed(DomainEvent):
    reason: str = ""


@dataclass(frozen=True)
class SessionStopRequested(DomainEvent):
    run_offline_pass: bool = True


@dataclass(frozen=True)
class SessionStopped(DomainEvent):
    stop_error: str = ""


@dataclass(frozen=True)
class OfflinePassStarted(DomainEvent):
    wav_path: str = ""


@dataclass(frozen=True)
class OfflinePassFinished(DomainEvent):
    out_txt: str = ""
    error: str = ""
