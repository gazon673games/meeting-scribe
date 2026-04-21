from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

from application.events.base import EventType, TypedEvent, init_event


@dataclass(frozen=True)
class CodexResultEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("ok", "profile", "cmd", "text", "dt_s")
    ok: bool
    profile: str
    cmd: str
    text: str
    dt_s: float

    def __init__(self, *, ok: bool, profile: str, cmd: str, text: str, dt_s: float, ts: Optional[float] = None):
        init_event(
            self,
            EventType.CODEX_RESULT,
            ts,
            ok=bool(ok),
            profile=str(profile),
            cmd=str(cmd),
            text=str(text),
            dt_s=float(dt_s),
        )


@dataclass(frozen=True)
class CodexFallbackStartedEvent(TypedEvent):
    record_fields: ClassVar[tuple[str, ...]] = ("profile", "cmd", "reason")
    profile: str
    cmd: str
    reason: str

    def __init__(self, *, profile: str, cmd: str, reason: str, ts: Optional[float] = None):
        init_event(
            self,
            EventType.CODEX_FALLBACK_STARTED,
            ts,
            profile=str(profile),
            cmd=str(cmd),
            reason=str(reason),
        )
