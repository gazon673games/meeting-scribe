from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from application.supervision import SupervisionReport, supervision_report


@dataclass(frozen=True)
class AssistantAttempt:
    label: str
    max_log_chars: Optional[int] = None
    timeout_s: Optional[int] = None
    fallback: bool = False


class AssistantFallbackSupervisor:
    component = "assistant"

    def build_attempts(
        self,
        *,
        max_log_chars: Optional[int],
        timeout_s: Optional[int],
        fallback_max_log_chars: Optional[int] = None,
        fallback_timeout_s: Optional[int] = None,
    ) -> List[AssistantAttempt]:
        attempts = [
            AssistantAttempt(
                label="primary",
                max_log_chars=max_log_chars,
                timeout_s=timeout_s,
                fallback=False,
            )
        ]
        if fallback_max_log_chars is not None or fallback_timeout_s is not None:
            attempts.append(
                AssistantAttempt(
                    label="fallback",
                    max_log_chars=fallback_max_log_chars if fallback_max_log_chars is not None else max_log_chars,
                    timeout_s=fallback_timeout_s if fallback_timeout_s is not None else timeout_s,
                    fallback=True,
                )
            )
        return attempts

    def success_report(self, attempt: AssistantAttempt, errors: List[str]) -> SupervisionReport:
        return supervision_report(
            component=self.component,
            active_attempt=attempt.label,
            fallback_used=bool(attempt.fallback),
            errors=errors,
        )

    def failure_report(self, errors: List[str]) -> SupervisionReport:
        return supervision_report(
            component=self.component,
            active_attempt="none",
            fallback_used=True,
            errors=errors,
            failed=True,
        )
