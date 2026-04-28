from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from application.codex_config import CodexProfile
from application.codex_prompting import build_codex_prompt
from assistant.application.provider import (
    ASSISTANT_PROVIDER_CODEX,
    AssistantExecutionSettings,
    AssistantProviderError,
    AssistantProviderInfo,
    AssistantProviderPort,
    AssistantProviderRequest,
    AssistantProviderResult,
    normalize_provider_id,
    provider_id_from_profile,
    result_from_error,
)
from transcription.application.transcript_context import TranscriptContextReader, trim_text_tail


@dataclass(frozen=True)
class AssistantRequestInput:
    user_text: str
    profile: CodexProfile
    project_root: Path
    human_log_path: Optional[Path]
    human_log_fh: Any
    max_log_chars: int
    answer_keyword: str
    execution_settings: AssistantExecutionSettings
    context_text: Optional[str] = None


class AssistantRequestUseCase:
    def __init__(
        self,
        providers: AssistantProviderPort | Iterable[AssistantProviderPort],
        context_reader: TranscriptContextReader,
    ) -> None:
        if isinstance(providers, Iterable) and not hasattr(providers, "run"):
            provider_list = list(providers)
        else:
            provider_list = [providers]  # type: ignore[list-item]
        self._providers = {
            normalize_provider_id(getattr(provider, "provider_id", ASSISTANT_PROVIDER_CODEX)): provider
            for provider in provider_list
        }
        self._context_reader = context_reader

    def execute(self, request: AssistantRequestInput) -> AssistantProviderResult:
        provider_id = provider_id_from_profile(request.profile)
        provider = self._providers.get(provider_id)
        if provider is None:
            return result_from_error(
                profile=request.profile,
                cmd=request.user_text,
                provider=provider_id,
                model=request.profile.model,
                error=AssistantProviderError(
                    code="provider_unavailable",
                    message=f"Assistant provider '{provider_id}' is not configured.",
                    retryable=False,
                    suggestion="Choose an available assistant profile or configure this provider in settings.",
                ),
            )

        if request.context_text is not None:
            log_text = trim_text_tail(request.context_text, max_chars=int(request.max_log_chars))
        else:
            log_text = self._context_reader.read_human_log_tail(
                project_root=Path(request.project_root),
                human_log_path=request.human_log_path,
                human_log_fh=request.human_log_fh,
                max_chars=int(request.max_log_chars),
            )
        prompt = build_codex_prompt(
            request.user_text,
            request.profile,
            log_text,
            answer_keyword=request.answer_keyword,
        )
        return provider.run(
            AssistantProviderRequest(
                prompt=prompt,
                profile=request.profile,
                original_cmd=request.user_text,
                project_root=Path(request.project_root),
                settings=request.execution_settings,
            )
        )

    def provider_statuses(self, settings: AssistantExecutionSettings) -> list[AssistantProviderInfo]:
        statuses: list[AssistantProviderInfo] = []
        for provider_id, provider in self._providers.items():
            status_fn = getattr(provider, "status", None)
            if callable(status_fn):
                try:
                    statuses.append(status_fn(settings))
                    continue
                except Exception as exc:
                    statuses.append(
                        AssistantProviderInfo(
                            id=provider_id,
                            label=str(getattr(provider, "provider_label", provider_id)),
                            available=False,
                            message=f"{type(exc).__name__}: {exc}",
                            error_code="provider_status_error",
                            retryable=True,
                        )
                    )
                    continue
            statuses.append(
                AssistantProviderInfo(
                    id=provider_id,
                    label=str(getattr(provider, "provider_label", provider_id)),
                    available=True,
                )
            )
        return statuses
