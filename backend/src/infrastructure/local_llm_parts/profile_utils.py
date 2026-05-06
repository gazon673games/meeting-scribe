from __future__ import annotations

from typing import Any

from assistant.application.provider import AssistantExecutionSettings, normalize_provider_id


def profile_for_provider(settings: AssistantExecutionSettings, provider_id: str) -> Any | None:
    selected = getattr(settings, "profile", None)
    if same_provider(selected, provider_id):
        return selected
    for profile in list(getattr(settings, "profiles", []) or []):
        if same_provider(profile, provider_id):
            return profile
    return None


def same_provider(profile: Any, provider_id: str) -> bool:
    if profile is None:
        return False
    return normalize_provider_id(getattr(profile, "provider_id", "")) == normalize_provider_id(provider_id)


def base_url(profile: Any, default: str) -> str:
    raw = str(getattr(profile, "base_url", "") or "").strip().rstrip("/")
    return raw or default


def status_timeout(settings: AssistantExecutionSettings) -> int:
    return min(3, max(1, int(getattr(settings, "timeout_s", 2) or 2)))


def temperature(profile: Any) -> float | None:
    raw = getattr(profile, "temperature", None)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return max(0.0, min(2.0, float(str(raw).replace(",", "."))))
    except Exception:
        return None


def max_tokens(profile: Any) -> int:
    try:
        return max(0, int(getattr(profile, "max_tokens", 0) or 0))
    except Exception:
        return 0


def ollama_options(profile: Any) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if (value := temperature(profile)) is not None:
        options["temperature"] = value
    if (value := max_tokens(profile)) > 0:
        options["num_predict"] = value
    return options


def auth_header(profile: Any) -> dict[str, str]:
    api_key = str(getattr(profile, "api_key", "") or "").strip()
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def ollama_models(data: dict[str, Any]) -> list[str]:
    return [
        str(item.get("name") or item.get("model") or "").strip()
        for item in data.get("models", [])
        if isinstance(item, dict) and str(item.get("name") or item.get("model") or "").strip()
    ]


def openai_models(data: dict[str, Any]) -> list[str]:
    return [
        str(item.get("id") or "").strip()
        for item in data.get("data", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    ]


def openai_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "").strip()
    return str(choices[0].get("text") or "").strip()
