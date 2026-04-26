from __future__ import annotations

from application.codex_config import CodexProfile


def build_codex_prompt(
    user_text: str,
    profile: CodexProfile,
    log_text: str,
    *,
    answer_keyword: str,
) -> str:
    cmd = str(user_text or "").strip()
    is_answer = cmd.upper() == str(answer_keyword).upper()
    model_hint = normalize_model_name(profile.model) or "default"
    effort_hint = normalize_reasoning_effort(profile.reasoning_effort) or "default"

    if is_answer:
        task = (profile.answer_prompt or "").strip()
        if not task:
            task = (
                "Command ANSWER: provide a fast answer for the latest question from context.\n"
                "Format:\n"
                "1) Short answer (1-3 sentences)\n"
                "2) Key points (up to 5)\n"
                "3) Optional clarification question"
            )
    else:
        task = cmd

    base = (
        "You are an assistant for realtime interview support. Be concise and practical.\n"
        f"Profile: {profile.label}\n"
        f"Model hint: {model_hint}\n\n"
        f"Reasoning effort hint: {effort_hint}\n\n"
        "Profile instructions:\n"
        f"{profile.prompt or '(empty)'}\n\n"
        "Task:\n"
        f"{task}\n\n"
        "Current session human-readable log:\n"
    )
    if log_text.strip():
        base += log_text.strip()
    else:
        base += "(log is empty)"
    return base


def normalize_model_name(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    sl = s.lower()
    effort_aliases = {
        "low",
        "medium",
        "high",
        "xhigh",
        "extra high",
        "extra_high",
        "extra-high",
    }
    if sl in effort_aliases:
        return ""
    return s


def normalize_reasoning_effort(raw: str) -> str:
    s = str(raw or "").strip().lower()
    if not s:
        return ""
    s = s.replace("(current)", "").strip()
    s_norm = s.replace("_", " ").replace("-", " ")
    s_compact = " ".join([x for x in s_norm.split() if x])
    mapping = {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "extra": "xhigh",
        "extra high": "xhigh",
        "very high": "xhigh",
    }
    return mapping.get(s_compact, "")
