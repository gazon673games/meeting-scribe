from __future__ import annotations

from dataclasses import dataclass

from assistant.application.provider import AssistantProviderError


@dataclass
class LocalLlmError(Exception):
    code: str
    message: str
    retryable: bool = True
    suggestion: str = ""
    status_code: int = 0

    def as_provider_error(self) -> AssistantProviderError:
        return AssistantProviderError(
            code=self.code,
            message=self.message,
            retryable=self.retryable,
            suggestion=self.suggestion,
        )
