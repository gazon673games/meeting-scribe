from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class AssistantAttempt:
    label: str
    max_log_chars: Optional[int] = None
    timeout_s: Optional[int] = None
    fallback: bool = False


class AssistantFallbackSupervisor:
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
