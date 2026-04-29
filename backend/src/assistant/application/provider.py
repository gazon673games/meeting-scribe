from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Protocol


ASSISTANT_PROVIDER_CODEX = "codex"
ASSISTANT_PROVIDER_OLLAMA = "ollama"
ASSISTANT_PROVIDER_OPENAI_LOCAL = "openai_local"


@dataclass(frozen=True)
class AssistantExecutionSettings:
    command_tokens: List[str]
    path_hints: List[str]
    proxy: str
    timeout_s: int
    project_root: Path | None = None
    profile: Any | None = None
    profiles: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class AssistantProviderError:
    code: str
    message: str
    retryable: bool = False
    suggestion: str = ""
    details: str = ""


@dataclass(frozen=True)
class AssistantProviderInfo:
    id: str
    label: str
    available: bool
    message: str = ""
    error_code: str = ""
    retryable: bool = False
    suggestion: str = ""
    models: list[str] = field(default_factory=list)
    auth_required: bool = False
    login_supported: bool = False
    local_home: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "available": bool(self.available),
            "message": self.message,
            "errorCode": self.error_code,
            "retryable": bool(self.retryable),
            "suggestion": self.suggestion,
            "models": list(self.models),
            "authRequired": bool(self.auth_required),
            "loginSupported": bool(self.login_supported),
            "localHome": self.local_home,
        }


@dataclass(frozen=True)
class AssistantProviderLoginResult:
    id: str
    label: str
    started: bool
    message: str = ""
    error_code: str = ""
    suggestion: str = ""
    local_home: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "started": bool(self.started),
            "message": self.message,
            "errorCode": self.error_code,
            "suggestion": self.suggestion,
            "localHome": self.local_home,
        }


@dataclass(frozen=True)
class AssistantProviderPingResult:
    id: str
    label: str
    ok: bool
    message: str = ""
    error_code: str = ""
    retryable: bool = False
    suggestion: str = ""
    status_code: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "ok": bool(self.ok),
            "message": self.message,
            "errorCode": self.error_code,
            "retryable": bool(self.retryable),
            "suggestion": self.suggestion,
            "statusCode": int(self.status_code),
        }


@dataclass(frozen=True)
class AssistantProviderRequest:
    prompt: str
    profile: Any
    original_cmd: str
    project_root: Path
    settings: AssistantExecutionSettings


@dataclass(frozen=True)
class AssistantProviderResult:
    ok: bool
    profile: str
    cmd: str
    text: str
    dt_s: float
    provider: str = ASSISTANT_PROVIDER_CODEX
    model: str = ""
    error_code: str = ""
    retryable: bool = False
    suggestion: str = ""
    details: str = ""

    @property
    def error(self) -> AssistantProviderError | None:
        if self.ok:
            return None
        return AssistantProviderError(
            code=str(self.error_code or "unknown_error"),
            message=str(self.text),
            retryable=bool(self.retryable),
            suggestion=str(self.suggestion or ""),
            details=str(self.details or ""),
        )


class AssistantProviderPort(Protocol):
    provider_id: str
    provider_label: str

    def run(self, request: AssistantProviderRequest) -> AssistantProviderResult:
        ...

    def status(self, settings: AssistantExecutionSettings) -> AssistantProviderInfo:
        ...


def provider_id_from_profile(profile: Any) -> str:
    value = getattr(profile, "provider_id", None) or getattr(profile, "provider", None) or ASSISTANT_PROVIDER_CODEX
    return normalize_provider_id(value)


def normalize_provider_id(value: Any) -> str:
    text = str(value or ASSISTANT_PROVIDER_CODEX).strip().lower()
    return text or ASSISTANT_PROVIDER_CODEX


def result_from_error(
    *,
    profile: Any,
    cmd: str,
    provider: str,
    error: AssistantProviderError,
    started_at: float | None = None,
    model: str = "",
) -> AssistantProviderResult:
    return AssistantProviderResult(
        ok=False,
        profile=str(getattr(profile, "label", "") or ""),
        cmd=str(cmd),
        text=str(error.message),
        dt_s=max(0.0, time.time() - started_at) if started_at is not None else 0.0,
        provider=normalize_provider_id(provider),
        model=str(model or getattr(profile, "model", "") or ""),
        error_code=str(error.code or "unknown_error"),
        retryable=bool(error.retryable),
        suggestion=str(error.suggestion or ""),
        details=str(error.details or ""),
    )
