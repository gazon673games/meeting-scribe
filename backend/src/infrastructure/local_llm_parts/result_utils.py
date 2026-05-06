from __future__ import annotations

import time
from typing import Any

from assistant.application.provider import (
    AssistantProviderError,
    AssistantProviderInfo,
    AssistantProviderPingResult,
    AssistantProviderResult,
    result_from_error,
)
from infrastructure.local_llm_parts.errors import LocalLlmError


def status_error(provider_id: str, label: str, error: LocalLlmError, base_url: str) -> AssistantProviderInfo:
    return AssistantProviderInfo(
        id=provider_id,
        label=label,
        available=False,
        message=f"{error.message} ({base_url})",
        error_code=error.code,
        retryable=error.retryable,
        suggestion=error.suggestion,
    )


def ping_ok(provider_id: str, label: str, message: str) -> AssistantProviderPingResult:
    return AssistantProviderPingResult(id=provider_id, label=label, ok=True, message=message, status_code=200)


def ping_error(provider_id: str, label: str, error: LocalLlmError) -> AssistantProviderPingResult:
    return AssistantProviderPingResult(
        id=provider_id,
        label=label,
        ok=False,
        message=error.message,
        error_code=error.code,
        retryable=error.retryable,
        suggestion=error.suggestion,
        status_code=error.status_code,
    )


def model_required(label: str) -> AssistantProviderError:
    return AssistantProviderError(
        code="model_required",
        message=f"{label} profile requires a model name.",
        suggestion="Set the assistant profile model to one of the models loaded in the local runtime.",
    )


def error_result(
    profile: Any,
    cmd: str,
    provider_id: str,
    model: str,
    error: AssistantProviderError,
    started: float,
) -> AssistantProviderResult:
    return result_from_error(
        profile=profile,
        cmd=cmd,
        provider=provider_id,
        model=model,
        error=error,
        started_at=started,
    )


def ok_result(
    profile: Any,
    cmd: str,
    provider_id: str,
    model: str,
    text: str,
    started: float,
) -> AssistantProviderResult:
    return AssistantProviderResult(
        ok=True,
        profile=str(getattr(profile, "label", "") or ""),
        cmd=str(cmd),
        text=text,
        dt_s=max(0.0, time.time() - started),
        provider=provider_id,
        model=model,
    )
