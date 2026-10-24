from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from application.codex_config import CodexSettings
from shared.domain.mappings import first_param

FAST_ANSWER_LOG_CHARS = 4000
FAST_ANSWER_TIMEOUT_S = 35
FAST_ANSWER_FALLBACK_LOG_CHARS = 2000
FAST_ANSWER_FALLBACK_TIMEOUT_S = 20
SUMMARY_LOG_CHARS = 200000
SUMMARY_TIMEOUT_S = 180
SUMMARY_REQUEST = (
    'Summarize the whole session context. '
    'Focus on decisions, open questions, risks, and the most useful next actions.'
)
ACTION_ITEMS_REQUEST = (
    'Extract concise action items from the current transcript. '
    'For each item include owner if clear, deadline if mentioned, and the next concrete step.'
)
RISK_CHECK_REQUEST = (
    'Review the current transcript for risks, gaps, and unclear points. '
    'Return the highest-impact risks first, then questions that should be clarified.'
)

_SUMMARY_ACTIONS = {
    'summary': (SUMMARY_REQUEST, 'summary'),
    'action_items': (ACTION_ITEMS_REQUEST, 'action items'),
    'risk_check': (RISK_CHECK_REQUEST, 'risk check'),
}


@dataclass(frozen=True)
class RequestPlan:
    request_text: str
    source_label: str
    limits: Mapping[str, int | None]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'limits', MappingProxyType(dict(self.limits)))


def build_request_plan(
    params: dict[str, Any], settings: CodexSettings
) -> RequestPlan:
    action = str(params.get('action', '') or '').strip().lower()
    raw_text = str(
        first_param(params, 'requestText', 'request_text', default='') or ''
    ).strip()
    if action == 'answer':
        return RequestPlan(
            request_text=str(settings.answer_keyword or 'ANSWER'),
            source_label='answer',
            limits={
                'max_log_chars': FAST_ANSWER_LOG_CHARS,
                'timeout_s': FAST_ANSWER_TIMEOUT_S,
                'fallback_max_log_chars': FAST_ANSWER_FALLBACK_LOG_CHARS,
                'fallback_timeout_s': FAST_ANSWER_FALLBACK_TIMEOUT_S,
            },
        )
    summary_action = _SUMMARY_ACTIONS.get(action)
    if summary_action is not None:
        request_text, source_label = summary_action
        return _summary_plan(request_text, source_label)
    if not raw_text:
        raise ValueError('Assistant request text is empty')
    return RequestPlan(
        request_text=raw_text,
        source_label=str(
            first_param(params, 'sourceLabel', 'source_label', default='you')
            or 'you'
        ),
        limits={
            'max_log_chars': optional_int(
                first_param(params, 'maxLogChars', 'max_log_chars')
            ),
            'timeout_s': optional_int(
                first_param(params, 'timeoutS', 'timeout_s')
            ),
            'fallback_max_log_chars': optional_int(
                first_param(
                    params, 'fallbackMaxLogChars', 'fallback_max_log_chars'
                )
            ),
            'fallback_timeout_s': optional_int(
                first_param(params, 'fallbackTimeoutS', 'fallback_timeout_s')
            ),
        },
    )


def resolve_context_text(
    params: dict[str, Any],
    settings: CodexSettings,
    transcript_supplier: Callable[[], str] | None,
) -> str | None:
    if 'contextText' in params:
        return str(params.get('contextText') or '')
    if str(settings.context_source or 'transcript') != 'transcript':
        return None
    if transcript_supplier is None:
        return ''
    return str(transcript_supplier())


def optional_int(raw: object) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError, OverflowError):
        return None


def _summary_plan(request_text: str, source_label: str) -> RequestPlan:
    return RequestPlan(
        request_text=request_text,
        source_label=source_label,
        limits={
            'max_log_chars': SUMMARY_LOG_CHARS,
            'timeout_s': SUMMARY_TIMEOUT_S,
            'fallback_max_log_chars': None,
            'fallback_timeout_s': None,
        },
    )
