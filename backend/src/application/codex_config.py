from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


DEFAULT_CODEX_PROXY = "http://127.0.0.1:10808"


@dataclass
class CodexProfile:
    id: str
    label: str
    prompt: str
    provider_id: str = "codex"
    model: str = ""
    reasoning_effort: str = ""
    codex_profile: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float | None = None
    max_tokens: int = 0
    answer_prompt: str = ""
    extra_args: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra_args is None:
            self.extra_args = []


@dataclass
class CodexSettings:
    enabled: bool = False
    proxy: str = DEFAULT_CODEX_PROXY
    answer_keyword: str = "ANSWER"
    timeout_s: int = 90
    max_log_chars: int = 24000
    command_tokens: List[str] = None  # type: ignore[assignment]
    path_hints: List[str] = None  # type: ignore[assignment]
    context_source: str = "transcript"
    console_expanded: bool = False
    selected_profile_id: str = ""
    profiles: List[CodexProfile] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.command_tokens is None:
            self.command_tokens = ["codex"]
        if self.path_hints is None:
            self.path_hints = []
        if self.profiles is None:
            self.profiles = default_codex_profiles()


def default_codex_profiles() -> List[CodexProfile]:
    return [
        CodexProfile(
            id="default",
            label="Default",
            model="",
            reasoning_effort="low",
            prompt=(
                "Support a realtime interview. Give short, practical, accurate responses from the session log."
            ),
            answer_prompt=(
                "Command ANSWER: provide a quick candidate response for the latest question. "
                "Format: 1) short answer 2) key points."
            ),
        )
    ]


def codex_profile_to_dict(profile: CodexProfile) -> Dict[str, Any]:
    return {
        "id": str(profile.id),
        "label": str(profile.label),
        "provider": str(profile.provider_id or "codex"),
        "prompt": str(profile.prompt),
        "model": str(profile.model),
        "reasoning_effort": str(profile.reasoning_effort),
        "codex_profile": str(profile.codex_profile),
        "base_url": str(profile.base_url),
        "api_key": str(profile.api_key),
        "temperature": profile.temperature if profile.temperature is not None else "",
        "max_tokens": int(profile.max_tokens or 0),
        "answer_prompt": str(profile.answer_prompt),
        "extra_args": [str(x) for x in list(profile.extra_args or []) if str(x).strip()],
    }


def parse_codex_profiles(raw_profiles: Any) -> List[CodexProfile]:
    out: List[CodexProfile] = []
    if isinstance(raw_profiles, list):
        for i, item in enumerate(raw_profiles):
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id") or f"profile_{i+1}").strip() or f"profile_{i+1}"
            label = str(item.get("label") or pid).strip() or pid
            prompt = str(item.get("prompt") or "").strip()
            provider_id = str(item.get("provider_id") or item.get("provider") or "codex").strip() or "codex"
            model = str(item.get("model") or "").strip()
            reasoning_effort = str(
                item.get("reasoning_effort")
                or item.get("reasoning")
                or item.get("reasoning_level")
                or ""
            ).strip()
            codex_profile = str(item.get("codex_profile") or "").strip()
            base_url = str(
                item.get("base_url")
                or item.get("baseUrl")
                or item.get("endpoint")
                or item.get("url")
                or ""
            ).strip()
            api_key = str(item.get("api_key") or item.get("apiKey") or "").strip()
            temperature = _optional_float(item.get("temperature"), lo=0.0, hi=2.0)
            max_tokens = _safe_int(item.get("max_tokens", item.get("maxTokens", 0)), default=0, lo=0, hi=200000)
            answer_prompt = str(item.get("answer_prompt") or "").strip()
            extra_args = parse_codex_list(item.get("extra_args", []))
            out.append(
                CodexProfile(
                    id=pid,
                    label=label,
                    prompt=prompt,
                    provider_id=provider_id,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    codex_profile=codex_profile,
                    base_url=base_url,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    answer_prompt=answer_prompt,
                    extra_args=extra_args,
                )
            )
    if not out:
        out = default_codex_profiles()
    return out


def parse_codex_command_tokens(raw_command: Any) -> List[str]:
    if isinstance(raw_command, list):
        tokens = [str(x).strip() for x in raw_command if str(x).strip()]
    elif isinstance(raw_command, str):
        s = raw_command.strip()
        tokens = [s] if s else []
    else:
        tokens = []
    return tokens or ["codex"]


def parse_codex_list(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def codex_command_config_value(command_tokens: List[str]) -> Any:
    if not command_tokens:
        return "codex"
    if len(command_tokens) == 1:
        return str(command_tokens[0])
    return [str(x) for x in command_tokens]


def parse_codex_settings(raw: Any) -> CodexSettings:
    codex = raw if isinstance(raw, dict) else {}
    proxy = codex.get("proxy", DEFAULT_CODEX_PROXY)
    if proxy is None:
        proxy = DEFAULT_CODEX_PROXY
    return CodexSettings(
        enabled=bool(codex.get("enabled", False)),
        proxy=str(proxy).strip(),
        answer_keyword=str(codex.get("answer_keyword", "ANSWER") or "ANSWER").strip() or "ANSWER",
        timeout_s=_safe_int(codex.get("timeout_s", 90), default=90, lo=10, hi=1200),
        max_log_chars=_safe_int(codex.get("max_log_chars", 24000), default=24000, lo=2000, hi=200000),
        command_tokens=parse_codex_command_tokens(codex.get("command", "codex")),
        path_hints=parse_codex_list(codex.get("path_hints", [])),
        context_source=str(codex.get("context_source", "transcript") or "transcript").strip() or "transcript",
        console_expanded=bool(codex.get("console_expanded", False)),
        selected_profile_id=str(codex.get("selected_profile", "")).strip(),
        profiles=parse_codex_profiles(codex.get("profiles", [])),
    )


def codex_settings_to_dict(settings: CodexSettings) -> Dict[str, Any]:
    return {
        "enabled": bool(settings.enabled),
        "proxy": str(settings.proxy),
        "answer_keyword": str(settings.answer_keyword),
        "timeout_s": int(settings.timeout_s),
        "max_log_chars": int(settings.max_log_chars),
        "command": codex_command_config_value(settings.command_tokens),
        "path_hints": [str(x) for x in settings.path_hints if str(x).strip()],
        "context_source": str(settings.context_source or "transcript"),
        "console_expanded": bool(settings.console_expanded),
        "selected_profile": str(settings.selected_profile_id or ""),
        "profiles": [codex_profile_to_dict(profile) for profile in settings.profiles],
    }


def _safe_int(raw: Any, *, default: int, lo: int, hi: int) -> int:
    try:
        value = int(str(raw).strip())
    except Exception:
        value = int(default)
    return max(int(lo), min(int(hi), int(value)))


def _optional_float(raw: Any, *, lo: float, hi: float) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        value = float(str(raw).strip().replace(",", "."))
    except Exception:
        return None
    return max(float(lo), min(float(hi), value))
