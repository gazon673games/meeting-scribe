from __future__ import annotations

from typing import Any

from application.codex_config import CodexProfile
from assistant.application.provider import (
    AssistantExecutionSettings,
    normalize_provider_id,
)


def profile_for_provider(
    settings: AssistantExecutionSettings, provider_id: str
) -> CodexProfile | None:
    selected = settings.profile
    if same_provider(selected, provider_id):
        return selected
    for profile in settings.profiles:
        if same_provider(profile, provider_id):
            return profile
    return None


def same_provider(profile: CodexProfile | None, provider_id: str) -> bool:
    if profile is None:
        return False
    return normalize_provider_id(profile.provider_id) == normalize_provider_id(
        provider_id
    )


def base_url(profile: CodexProfile | None, default: str) -> str:
    raw = str(profile.base_url or '').strip().rstrip('/') if profile else ''
    return raw or default


def status_timeout(settings: AssistantExecutionSettings) -> int:
    return min(3, max(1, int(settings.timeout_s or 2)))


def temperature(profile: CodexProfile | None) -> float | None:
    raw = profile.temperature if profile is not None else None
    if raw is None or str(raw).strip() == '':
        return None
    try:
        return max(0.0, min(2.0, float(str(raw).replace(',', '.'))))
    except (TypeError, ValueError, OverflowError):
        return None


def max_tokens(profile: CodexProfile | None) -> int:
    if profile is None:
        return 0
    try:
        return max(0, int(profile.max_tokens or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def ollama_options(profile: CodexProfile | None) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if (value := temperature(profile)) is not None:
        options['temperature'] = value
    if (value := max_tokens(profile)) > 0:
        options['num_predict'] = value
    return options


def auth_header(profile: CodexProfile | None) -> dict[str, str]:
    api_key = str(profile.api_key or '').strip() if profile is not None else ''
    return {'Authorization': f'Bearer {api_key}'} if api_key else {}


def ollama_models(data: dict[str, Any]) -> list[str]:
    models = data.get('models')
    if not isinstance(models, list):
        return []
    return [
        str(item.get('name') or item.get('model') or '').strip()
        for item in models
        if isinstance(item, dict)
        and str(item.get('name') or item.get('model') or '').strip()
    ]


def openai_models(data: dict[str, Any]) -> list[str]:
    models = data.get('data')
    if not isinstance(models, list):
        return []
    return [
        str(item.get('id') or '').strip()
        for item in models
        if isinstance(item, dict) and str(item.get('id') or '').strip()
    ]


def openai_text(data: dict[str, Any]) -> str:
    choices = data.get('choices')
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
    ):
        return ''
    message = choices[0].get('message')
    if isinstance(message, dict):
        return str(message.get('content') or '').strip()
    return str(choices[0].get('text') or '').strip()
