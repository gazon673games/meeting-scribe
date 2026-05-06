from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from application.codex_config import CodexSettings

FAST_ANSWER_LOG_CHARS = 4000
FAST_ANSWER_TIMEOUT_S = 35
FAST_ANSWER_FALLBACK_LOG_CHARS = 2000
FAST_ANSWER_FALLBACK_TIMEOUT_S = 20
SUMMARY_LOG_CHARS = 200000
SUMMARY_TIMEOUT_S = 180
SUMMARY_REQUEST = (
    "Summarize the whole session context. "
    "Focus on decisions, open questions, risks, and the most useful next actions."
)
ACTION_ITEMS_REQUEST = (
    "Extract concise action items from the current transcript. "
    "For each item include owner if clear, deadline if mentioned, and the next concrete step."
)
RISK_CHECK_REQUEST = (
    "Review the current transcript for risks, gaps, and unclear points. "
    "Return the highest-impact risks first, then questions that should be clarified."
)


@dataclass(frozen=True)
class RequestPlan:
    request_text: str
    source_label: str
    limits: Dict[str, Optional[int]]


def build_request_plan(params: Dict[str, Any], settings: CodexSettings) -> RequestPlan:
    action = str(params.get("action", "") or "").strip().lower()
    raw_text = str(params.get("requestText", params.get("request_text", "")) or "").strip()
    if action == "answer":
        return RequestPlan(
            request_text=str(settings.answer_keyword or "ANSWER"),
            source_label="answer",
            limits={
                "max_log_chars": FAST_ANSWER_LOG_CHARS,
                "timeout_s": FAST_ANSWER_TIMEOUT_S,
                "fallback_max_log_chars": FAST_ANSWER_FALLBACK_LOG_CHARS,
                "fallback_timeout_s": FAST_ANSWER_FALLBACK_TIMEOUT_S,
            },
        )
    if action == "summary":
        return _summary_plan(SUMMARY_REQUEST, "summary")
    if action == "action_items":
        return _summary_plan(ACTION_ITEMS_REQUEST, "action items")
    if action == "risk_check":
        return _summary_plan(RISK_CHECK_REQUEST, "risk check")
    if not raw_text:
        raise ValueError("Assistant request text is empty")
    return RequestPlan(
        request_text=raw_text,
        source_label=str(params.get("sourceLabel", params.get("source_label", "you")) or "you"),
        limits={
            "max_log_chars": optional_int(params.get("maxLogChars", params.get("max_log_chars"))),
            "timeout_s": optional_int(params.get("timeoutS", params.get("timeout_s"))),
            "fallback_max_log_chars": optional_int(
                params.get("fallbackMaxLogChars", params.get("fallback_max_log_chars"))
            ),
            "fallback_timeout_s": optional_int(params.get("fallbackTimeoutS", params.get("fallback_timeout_s"))),
        },
    )


def resolve_context_text(
    params: Dict[str, Any],
    settings: CodexSettings,
    transcript_supplier: Any,  # noqa: ANN401
) -> Optional[str]:
    if "contextText" in params:
        return str(params.get("contextText") or "")
    if str(settings.context_source or "transcript") != "transcript":
        return None
    if transcript_supplier is None:
        return ""
    return str(transcript_supplier())


def optional_int(raw: object) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _summary_plan(request_text: str, source_label: str) -> RequestPlan:
    return RequestPlan(
        request_text=request_text,
        source_label=source_label,
        limits={
            "max_log_chars": SUMMARY_LOG_CHARS,
            "timeout_s": SUMMARY_TIMEOUT_S,
            "fallback_max_log_chars": None,
            "fallback_timeout_s": None,
        },
    )
