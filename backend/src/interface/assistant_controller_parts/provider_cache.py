from __future__ import annotations

from typing import Any, Dict

from application.codex_config import CodexProfile, CodexSettings
from assistant.application.provider import ASSISTANT_PROVIDER_CODEX


def provider_snapshot_fields(provider: Dict[str, Any] | None) -> Dict[str, Any]:
    if provider is None:
        return {
            "providerAvailable": False,
            "providerId": "",
            "providerMessage": "",
            "providerErrorCode": "",
            "providerSuggestion": "",
            "providerAuthRequired": False,
            "providerLoginSupported": False,
            "providerLocalHome": "",
        }
    return {
        "providerAvailable": bool(provider.get("available", False)),
        "providerId": str(provider.get("id", "")),
        "providerMessage": str(provider.get("message", "")),
        "providerErrorCode": str(provider.get("errorCode", "")),
        "providerSuggestion": str(provider.get("suggestion", "")),
        "providerAuthRequired": bool(provider.get("authRequired", False)),
        "providerLoginSupported": bool(provider.get("loginSupported", False)),
        "providerLocalHome": str(provider.get("localHome", "")),
    }


def provider_for_profile(
    providers: list[Dict[str, Any]],
    profile: CodexProfile | None,
) -> Dict[str, Any] | None:
    if not providers:
        return None
    wanted = str(getattr(profile, "provider_id", "") or ASSISTANT_PROVIDER_CODEX).strip().lower()
    for provider in providers:
        if str(provider.get("id", "")).strip().lower() == wanted:
            return provider
    return None


def provider_cache_key(settings: CodexSettings) -> tuple:
    return (
        bool(settings.enabled),
        str(settings.proxy or ""),
        tuple(settings.command_tokens or []),
        tuple(settings.path_hints or []),
        tuple(
            (
                str(profile.id),
                str(profile.provider_id),
                str(profile.model),
                str(profile.base_url),
                str(profile.codex_profile),
            )
            for profile in list(settings.profiles or [])
        ),
    )
