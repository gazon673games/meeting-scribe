from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from application.codex_config import CodexProfile
from application.codex_prompting import build_codex_prompt
from assistant.application.provider import (
    ASSISTANT_PROVIDER_CODEX,
    AssistantExecutionSettings,
    AssistantProviderError,
    AssistantProviderInfo,
    AssistantProviderLoginResult,
    AssistantProviderPingResult,
    AssistantProviderPort,
    AssistantProviderRequest,
    AssistantProviderResult,
    normalize_provider_id,
    provider_id_from_profile,
    result_from_error,
)
from transcription.application.transcript_context import (
    TranscriptContextReader,
    trim_text_tail,
)


@dataclass(frozen=True)
class AssistantRequestInput:
    user_text: str
    profile: CodexProfile
    project_root: Path
    human_log_path: Path | None
    human_log_fh: TextIO | None
    max_log_chars: int
    answer_keyword: str
    execution_settings: AssistantExecutionSettings
    context_text: str | None = None


@dataclass(frozen=True)
class _ProviderBinding:
    label: str
    run: Callable[[AssistantProviderRequest], AssistantProviderResult]
    status: Callable[[AssistantExecutionSettings], AssistantProviderInfo] | None
    login: Callable[..., AssistantProviderLoginResult] | None
    ping: (
        Callable[[AssistantExecutionSettings], AssistantProviderPingResult]
        | None
    )


def _optional_method(
    provider: AssistantProviderPort, name: str
) -> Callable | None:
    method = getattr(provider, name, None)
    return method if callable(method) else None


def _bind_provider(provider: AssistantProviderPort) -> _ProviderBinding:
    provider_id = normalize_provider_id(
        getattr(provider, 'provider_id', ASSISTANT_PROVIDER_CODEX)
    )
    return _ProviderBinding(
        label=str(getattr(provider, 'provider_label', provider_id)),
        run=provider.run,
        status=_optional_method(provider, 'status'),
        login=_optional_method(provider, 'start_login'),
        ping=_optional_method(provider, 'ping'),
    )


class AssistantRequestUseCase:
    def __init__(
        self,
        providers: AssistantProviderPort | Iterable[AssistantProviderPort],
        context_reader: TranscriptContextReader,
    ) -> None:
        if isinstance(providers, Iterable) and not hasattr(providers, 'run'):
            provider_list = list(providers)
        else:
            provider_list = [providers]  # type: ignore[list-item]
        self._providers = {
            normalize_provider_id(
                getattr(provider, 'provider_id', ASSISTANT_PROVIDER_CODEX)
            ): _bind_provider(provider)
            for provider in provider_list
        }
        self._context_reader = context_reader

    def execute(
        self, request: AssistantRequestInput
    ) -> AssistantProviderResult:
        provider_id = provider_id_from_profile(request.profile)
        provider = self._providers.get(provider_id)
        if provider is None:
            return result_from_error(
                profile=request.profile,
                cmd=request.user_text,
                provider=provider_id,
                model=request.profile.model,
                error=AssistantProviderError(
                    code='provider_unavailable',
                    message=f"Assistant provider '{provider_id}' is not configured.",
                    retryable=False,
                    suggestion='Choose an available assistant profile or configure this provider in settings.',
                ),
            )

        if request.context_text is not None:
            log_text = trim_text_tail(
                request.context_text, max_chars=int(request.max_log_chars)
            )
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

    def provider_statuses(
        self, settings: AssistantExecutionSettings
    ) -> list[AssistantProviderInfo]:
        statuses: list[AssistantProviderInfo] = []
        for provider_id, provider in self._providers.items():
            status_fn = provider.status
            if status_fn is not None:
                try:
                    statuses.append(status_fn(settings))
                    continue
                except Exception as error:
                    statuses.append(
                        AssistantProviderInfo(
                            id=provider_id,
                            label=str(provider.label),
                            available=False,
                            message=f'{type(error).__name__}: {error}',
                            error_code='provider_status_error',
                            retryable=True,
                        )
                    )
                    continue
            statuses.append(
                AssistantProviderInfo(
                    id=provider_id,
                    label=str(provider.label),
                    available=True,
                )
            )
        return statuses

    def start_provider_login(
        self,
        provider_id: str,
        settings: AssistantExecutionSettings,
        *,
        device_auth: bool = False,
    ) -> AssistantProviderLoginResult:
        normalized = normalize_provider_id(provider_id)
        provider = self._providers.get(normalized)
        if provider is None:
            return AssistantProviderLoginResult(
                id=normalized,
                label=normalized,
                started=False,
                error_code='provider_unavailable',
                message=f"Assistant provider '{normalized}' is not configured.",
                suggestion='Choose an available assistant profile or configure this provider in settings.',
            )

        login_fn = provider.login
        if login_fn is None:
            return AssistantProviderLoginResult(
                id=normalized,
                label=provider.label,
                started=False,
                error_code='login_not_supported',
                message=f"Assistant provider '{normalized}' does not support interactive login.",
            )

        try:
            return login_fn(settings, device_auth=bool(device_auth))
        except Exception as error:
            return AssistantProviderLoginResult(
                id=normalized,
                label=provider.label,
                started=False,
                error_code='login_start_failed',
                message=f'{type(error).__name__}: {error}',
                suggestion='Run codex login manually in a terminal if the login window did not open.',
            )

    def ping_provider(
        self, provider_id: str, settings: AssistantExecutionSettings
    ) -> AssistantProviderPingResult:
        normalized = normalize_provider_id(provider_id)
        provider = self._providers.get(normalized)
        if provider is None:
            return AssistantProviderPingResult(
                id=normalized,
                label=normalized,
                ok=False,
                error_code='provider_unavailable',
                message=f"Assistant provider '{normalized}' is not configured.",
                suggestion='Choose an available assistant profile or configure this provider in settings.',
            )

        ping_fn = provider.ping
        if ping_fn is None:
            return AssistantProviderPingResult(
                id=normalized,
                label=provider.label,
                ok=False,
                error_code='ping_not_supported',
                message=f"Assistant provider '{normalized}' does not support connectivity ping.",
            )

        try:
            return ping_fn(settings)
        except Exception as error:
            return AssistantProviderPingResult(
                id=normalized,
                label=provider.label,
                ok=False,
                error_code='ping_failed',
                message=f'{type(error).__name__}: {error}',
                retryable=True,
                suggestion='Check internet access, DNS, firewall, or proxy settings.',
            )
