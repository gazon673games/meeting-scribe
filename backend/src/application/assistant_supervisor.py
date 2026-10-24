from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from application.supervision import SupervisionReport, supervision_report


@dataclass(frozen=True)
class AssistantAttempt:
    label: str
    max_log_chars: int | None = None
    timeout_s: int | None = None
    fallback: bool = False


class AssistantFallbackPolicy(Protocol):
    def build_attempts(
        self,
        *,
        max_log_chars: int | None,
        timeout_s: int | None,
        fallback_max_log_chars: int | None = None,
        fallback_timeout_s: int | None = None,
    ) -> list[AssistantAttempt]: ...

    def success_report(
        self, attempt: AssistantAttempt, errors: list[str]
    ) -> SupervisionReport: ...

    def failure_report(self, errors: list[str]) -> SupervisionReport: ...


COMPONENT = 'assistant'


def build_attempts(
    *,
    max_log_chars: int | None,
    timeout_s: int | None,
    fallback_max_log_chars: int | None = None,
    fallback_timeout_s: int | None = None,
) -> list[AssistantAttempt]:
    attempts = [
        AssistantAttempt(
            label='primary',
            max_log_chars=max_log_chars,
            timeout_s=timeout_s,
            fallback=False,
        )
    ]
    if fallback_max_log_chars is not None or fallback_timeout_s is not None:
        attempts.append(
            AssistantAttempt(
                label='fallback',
                max_log_chars=fallback_max_log_chars
                if fallback_max_log_chars is not None
                else max_log_chars,
                timeout_s=fallback_timeout_s
                if fallback_timeout_s is not None
                else timeout_s,
                fallback=True,
            )
        )
    return attempts


def success_report(
    attempt: AssistantAttempt, errors: list[str]
) -> SupervisionReport:
    return supervision_report(
        component=COMPONENT,
        active_attempt=attempt.label,
        fallback_used=bool(attempt.fallback),
        errors=errors,
    )


def failure_report(errors: list[str]) -> SupervisionReport:
    return supervision_report(
        component=COMPONENT,
        active_attempt='none',
        fallback_used=True,
        errors=errors,
        failed=True,
    )


class AssistantFallbackSupervisor:
    """Compatibility facade; runtime code uses the module functions."""

    component = COMPONENT
    build_attempts = staticmethod(build_attempts)
    success_report = staticmethod(success_report)
    failure_report = staticmethod(failure_report)
