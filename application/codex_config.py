from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


DEFAULT_CODEX_PROXY = "http://127.0.0.1:10808"


@dataclass
class CodexProfile:
    id: str
    label: str
    prompt: str
    model: str = ""
    reasoning_effort: str = ""
    codex_profile: str = ""
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
        "prompt": str(profile.prompt),
        "model": str(profile.model),
        "reasoning_effort": str(profile.reasoning_effort),
        "codex_profile": str(profile.codex_profile),
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
            model = str(item.get("model") or "").strip()
            reasoning_effort = str(
                item.get("reasoning_effort")
                or item.get("reasoning")
                or item.get("reasoning_level")
                or ""
            ).strip()
            codex_profile = str(item.get("codex_profile") or "").strip()
            answer_prompt = str(item.get("answer_prompt") or "").strip()
            extra_args = parse_codex_list(item.get("extra_args", []))
            out.append(
                CodexProfile(
                    id=pid,
                    label=label,
                    prompt=prompt,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    codex_profile=codex_profile,
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
    return CodexSettings(
        enabled=bool(codex.get("enabled", False)),
        proxy=str(codex.get("proxy", DEFAULT_CODEX_PROXY) or DEFAULT_CODEX_PROXY).strip(),
        answer_keyword=str(codex.get("answer_keyword", "ANSWER") or "ANSWER").strip() or "ANSWER",
        timeout_s=_safe_int(codex.get("timeout_s", 90), default=90, lo=10, hi=1200),
        max_log_chars=_safe_int(codex.get("max_log_chars", 24000), default=24000, lo=2000, hi=200000),
        command_tokens=parse_codex_command_tokens(codex.get("command", "codex")),
        path_hints=parse_codex_list(codex.get("path_hints", [])),
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
