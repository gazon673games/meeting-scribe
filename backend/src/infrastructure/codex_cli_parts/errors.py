from __future__ import annotations

from assistant.application.provider import AssistantProviderError, AssistantProviderInfo


ERR_NOT_FOUND = (
    "codex executable not found. "
    "Set codex.command in config.json "
    "(e.g. 'C:/Users/<you>/AppData/Roaming/npm/codex.cmd' or full codex.exe path)."
)
ERR_NOT_FOUND_RUNTIME = (
    "codex executable not found at runtime. "
    "Try setting codex.command in config.json to an explicit path."
)


def codex_not_found_error(*, runtime: bool = False) -> AssistantProviderError:
    return AssistantProviderError(
        code="codex_not_found",
        message=ERR_NOT_FOUND_RUNTIME if runtime else ERR_NOT_FOUND,
        retryable=False,
        suggestion=(
            "Set codex.command to an explicit codex executable path."
            if runtime else
            "Install Codex CLI or set codex.command to the full executable path."
        ),
    )


def status_error_text(text: str, *, source: str, returncode: int) -> str:
    clean = str(text or "").strip()
    if clean:
        return f"{clean}\n(source={source})"
    return f"codex login status failed with code {returncode}\n(source={source})"


def classify_codex_error(message: str) -> AssistantProviderError:
    text = str(message or "").strip()
    lower = text.lower()
    if any(marker in lower for marker in ("rate limit", "rate_limit", "too many requests", " 429", "(429", "status 429")):
        return AssistantProviderError(
            code="rate_limited",
            message=text,
            retryable=True,
            suggestion="Wait a little or switch to another assistant profile/model.",
        )
    if any(
        marker in lower
        for marker in (
            "login",
            "auth",
            "unauthorized",
            "forbidden",
            "api key",
            "not logged in",
            "logged out",
            "not authenticated",
            "401",
            "403",
        )
    ):
        return AssistantProviderError(
            code="auth_error",
            message=text,
            retryable=False,
            suggestion="Authorize the local Codex profile or check API credentials and try again.",
        )
    if any(
        marker in lower
        for marker in (
            "enotfound",
            "eai_again",
            "dns",
            "network",
            "connection",
            "connect",
            "proxy",
            "timed out",
            "timeout",
            "tls",
            "ssl",
        )
    ):
        return AssistantProviderError(
            code="network_error",
            message=text,
            retryable=True,
            suggestion="Check internet access or proxy settings.",
        )
    if "model" in lower and any(marker in lower for marker in ("not found", "unavailable", "unsupported", "unknown")):
        return AssistantProviderError(
            code="model_unavailable",
            message=text,
            retryable=False,
            suggestion="Choose another Codex model/profile.",
        )
    return AssistantProviderError(
        code="provider_crash",
        message=text,
        retryable=True,
        suggestion="Retry the request; if it repeats, check the assistant console output.",
    )


def provider_info_from_error(
    error: AssistantProviderError,
    *,
    provider_id: str,
    provider_label: str,
    local_home: str = "",
) -> AssistantProviderInfo:
    return AssistantProviderInfo(
        id=provider_id,
        label=provider_label,
        available=False,
        message=error.message,
        error_code=error.code,
        retryable=error.retryable,
        suggestion=error.suggestion,
        auth_required=error.code == "auth_error",
        login_supported=error.code != "codex_not_found",
        local_home=local_home,
    )
