from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from application.codex_config import CodexProfile
from assistant.application.provider import (
    AssistantProviderInfo,
    AssistantProviderLoginResult,
    AssistantProviderPingResult,
)
from assistant.application.service import AssistantRuntimeOptions


class ProviderLogin(Protocol):
    def __call__(
        self,
        provider_id: str,
        *,
        options: AssistantRuntimeOptions,
        device_auth: bool = False,
    ) -> AssistantProviderLoginResult: ...


class ProviderPing(Protocol):
    def __call__(
        self,
        provider_id: str,
        *,
        options: AssistantRuntimeOptions,
        profile: CodexProfile | None = None,
    ) -> AssistantProviderPingResult: ...


class ProviderStatuses(Protocol):
    def __call__(
        self, *, options: AssistantRuntimeOptions
    ) -> list[AssistantProviderInfo]: ...


@dataclass(frozen=True)
class ProviderOperations:
    login: ProviderLogin | None
    ping: ProviderPing | None
    statuses: ProviderStatuses | None


def provider_operations(service: object) -> ProviderOperations:
    # Normalize optional methods once at the compatibility boundary.
    login = getattr(service, 'start_provider_login', None)
    ping = getattr(service, 'ping_provider', None)
    statuses = getattr(service, 'provider_statuses', None)
    return ProviderOperations(
        login=login if callable(login) else None,
        ping=ping if callable(ping) else None,
        statuses=statuses if callable(statuses) else None,
    )
