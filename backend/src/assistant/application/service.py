from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

from application.assistant_supervisor import AssistantFallbackSupervisor
from application.assistant_use_case import AssistantRequestInput, AssistantRequestUseCase
from application.commands import InvokeAssistantCommand
from application.event_types import CodexFallbackStartedEvent, CodexResultEvent
from application.supervision import SupervisionReport
from assistant.application.provider import (
    AssistantExecutionSettings,
    AssistantProviderInfo,
    AssistantProviderLoginResult,
    AssistantProviderPingResult,
)


@dataclass(frozen=True)
class AssistantRuntimeOptions:
    project_root: Path
    default_max_log_chars: int
    answer_keyword: str
    command_tokens: list[str]
    path_hints: list[str]
    proxy: str
    default_timeout_s: int
    profiles: list[Any] | None = None


class AssistantApplicationService:
    def __init__(
        self,
        use_case: AssistantRequestUseCase,
        *,
        supervisor: Optional[AssistantFallbackSupervisor] = None,
    ) -> None:
        self._use_case = use_case
        self._supervisor = supervisor or AssistantFallbackSupervisor()
        self.last_supervision_report: Optional[SupervisionReport] = None

    def execute(
        self,
        command: InvokeAssistantCommand,
        *,
        options: AssistantRuntimeOptions,
        publish_event: Callable[[object], None],
    ) -> None:
        attempts = self._supervisor.build_attempts(
            max_log_chars=command.max_log_chars,
            timeout_s=command.timeout_s,
            fallback_max_log_chars=command.fallback_max_log_chars,
            fallback_timeout_s=command.fallback_timeout_s,
        )
        result = None
        first_error = ""
        errors: List[str] = []

        for attempt in attempts:
            if attempt.fallback:
                publish_event(
                    CodexFallbackStartedEvent(
                        profile=command.profile.label,
                        cmd=command.request_text,
                        reason=first_error or "primary attempt failed",
                    )
                )

            result = self._use_case.execute(
                AssistantRequestInput(
                    user_text=command.request_text,
                    profile=command.profile,
                    project_root=Path(options.project_root),
                    human_log_path=command.human_log_path,
                    human_log_fh=command.human_log_fh,
                    max_log_chars=int(
                        attempt.max_log_chars
                        if attempt.max_log_chars is not None
                        else options.default_max_log_chars
                    ),
                    answer_keyword=options.answer_keyword,
                    context_text=command.context_text,
                    execution_settings=AssistantExecutionSettings(
                        command_tokens=list(options.command_tokens),
                        path_hints=list(options.path_hints),
                        proxy=str(options.proxy or ""),
                        timeout_s=int(
                            attempt.timeout_s if attempt.timeout_s is not None else options.default_timeout_s
                        ),
                        project_root=Path(options.project_root),
                        profile=command.profile,
                        profiles=list(options.profiles or []),
                    ),
                )
            )
            if result.ok:
                self.last_supervision_report = self._supervisor.success_report(attempt, errors)
                break
            first_error = result.text
            errors.append(f"{attempt.label}: {result.text}")

        if result is None:
            return
        if not result.ok:
            self.last_supervision_report = self._supervisor.failure_report(errors)

        publish_event(
            CodexResultEvent(
                ok=bool(result.ok),
                profile=result.profile,
                cmd=result.cmd,
                text=result.text,
                dt_s=float(result.dt_s),
                provider=result.provider,
                model=result.model,
                error_code=result.error_code,
                retryable=result.retryable,
                suggestion=result.suggestion,
                details=result.details,
            )
        )

    def provider_statuses(self, *, options: AssistantRuntimeOptions) -> list[AssistantProviderInfo]:
        return self._use_case.provider_statuses(
            AssistantExecutionSettings(
                command_tokens=list(options.command_tokens),
                path_hints=list(options.path_hints),
                proxy=str(options.proxy or ""),
                timeout_s=int(options.default_timeout_s),
                project_root=Path(options.project_root),
                profiles=list(options.profiles or []),
            )
        )

    def start_provider_login(
        self,
        provider_id: str,
        *,
        options: AssistantRuntimeOptions,
        device_auth: bool = False,
    ) -> AssistantProviderLoginResult:
        return self._use_case.start_provider_login(
            provider_id,
            AssistantExecutionSettings(
                command_tokens=list(options.command_tokens),
                path_hints=list(options.path_hints),
                proxy=str(options.proxy or ""),
                timeout_s=int(options.default_timeout_s),
                project_root=Path(options.project_root),
                profiles=list(options.profiles or []),
            ),
            device_auth=bool(device_auth),
        )

    def ping_provider(
        self,
        provider_id: str,
        *,
        options: AssistantRuntimeOptions,
        profile: Any | None = None,
    ) -> AssistantProviderPingResult:
        return self._use_case.ping_provider(
            provider_id,
            AssistantExecutionSettings(
                command_tokens=list(options.command_tokens),
                path_hints=list(options.path_hints),
                proxy=str(options.proxy or ""),
                timeout_s=int(options.default_timeout_s),
                project_root=Path(options.project_root),
                profile=profile,
                profiles=list(options.profiles or []),
            ),
        )
