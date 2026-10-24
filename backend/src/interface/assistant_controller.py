from __future__ import annotations

import logging
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from application.codex_config import (
    CodexProfile,
    CodexSettings,
    parse_codex_profiles,
    parse_codex_settings,
)
from application.commands import InvokeAssistantCommand
from application.event_types import (
    CodexFallbackStartedEvent,
    CodexResultEvent,
    event_to_record,
)
from assistant.application.local_models import LocalModelRuntimePort
from assistant.application.provider import ASSISTANT_PROVIDER_CODEX
from assistant.application.service import (
    AssistantApplicationService,
    AssistantRuntimeOptions,
)
from assistant.domain.aggregate import AssistantJobAggregate
from infrastructure import local_model_runtime
from interface.assistant_controller_parts import (
    build_request_plan,
    provider_cache_key,
    provider_for_profile,
    provider_snapshot_fields,
    resolve_context_text,
)
from interface.assistant_controller_parts.provider_operations import (
    ProviderOperations,
    ProviderPing,
    provider_operations,
)
from interface.session_controller import HeadlessSessionController
from interface.session_controller_parts.types import EventSink
from settings.application.config_repository import ConfigRepository
from shared.domain.mappings import first_param

_logger = logging.getLogger(__name__)


@dataclass
class AssistantController:
    project_root: Path
    config_repository: ConfigRepository
    assistant_service: AssistantApplicationService
    session_controller: HeadlessSessionController | None = None
    event_sink: EventSink | None = None
    local_models: LocalModelRuntimePort = field(
        default=local_model_runtime, kw_only=True
    )
    _provider_operations: ProviderOperations = field(init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _job_state: AssistantJobAggregate = field(
        default_factory=AssistantJobAggregate
    )
    _last_response: dict[str, Any] = field(default_factory=dict)
    _last_error: str = ''
    _last_request: dict[str, Any] = field(default_factory=dict)
    _worker: threading.Thread | None = field(
        default=None, init=False, repr=False
    )
    _closing: bool = field(default=False, init=False, repr=False)
    _ping_workers: set[threading.Thread] = field(
        default_factory=set, init=False, repr=False
    )
    _provider_cache_key: tuple = field(
        default_factory=tuple, init=False, repr=False
    )
    _provider_cache_ts: float = field(default=0.0, init=False, repr=False)
    _provider_cache_records: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self._provider_operations = provider_operations(self.assistant_service)

    def set_event_sink(self, event_sink: EventSink | None) -> None:
        with self._lock:
            self.event_sink = event_sink

    def snapshot(self) -> dict[str, Any]:
        settings = self._settings()
        selected = (
            self._selected_profile(settings) if settings.profiles else None
        )
        providers = self._provider_records(settings)
        selected_provider = (
            provider_for_profile(providers, selected)
            if selected is not None
            else None
        )
        with self._lock:
            return {
                'enabled': bool(settings.enabled),
                **provider_snapshot_fields(selected_provider),
                'busy': bool(self._job_state.is_busy),
                'fallback': bool(self._job_state.is_fallback),
                'selectedProfileId': selected.id
                if selected is not None
                else '',
                'profiles': [
                    _profile_record(profile) for profile in settings.profiles
                ],
                'providers': providers,
                'lastResponse': deepcopy(self._last_response),
                'lastError': self._last_error,
                'lastRequest': deepcopy(self._last_request),
            }

    def invoke(self, params: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings()
        if not settings.enabled:
            raise RuntimeError('Assistant is disabled in config')
        profile = self._select_profile(
            settings,
            str(
                first_param(params, 'profileId', 'profile_id', default='') or ''
            ),
            strict=True,
        )
        plan = build_request_plan(params, settings)
        request_text = plan.request_text
        source_label = plan.source_label
        limits = plan.limits
        context_text = resolve_context_text(
            params=params,
            settings=settings,
            transcript_supplier=self.session_controller.transcript_text
            if self.session_controller is not None
            else None,
        )
        context_label = (
            'current transcript' if context_text else 'no transcript'
        )

        command = InvokeAssistantCommand(
            profile=profile,
            request_text=request_text,
            source_label=source_label,
            context_source=str(settings.context_source or 'transcript'),
            context_label=context_label,
            context_text=context_text,
            max_log_chars=limits.get('max_log_chars'),
            timeout_s=limits.get('timeout_s'),
            fallback_max_log_chars=limits.get('fallback_max_log_chars'),
            fallback_timeout_s=limits.get('fallback_timeout_s'),
        )

        with self._lock:
            if self._closing:
                raise RuntimeError('Assistant is shutting down')
            if self._job_state.is_busy:
                raise RuntimeError('Assistant is already running')
            self._job_state.begin(
                profile=profile.label, source_label=source_label
            )
            self._last_error = ''
            self._last_request = {
                'profile': profile.label,
                'profileId': profile.id,
                'sourceLabel': source_label,
                'requestText': request_text,
                'contextLabel': context_label,
                'ts': time.time(),
            }

        self._emit('assistant_started', dict(self._last_request))
        worker = threading.Thread(
            target=self._run_worker,
            args=(command, settings),
            name='electron-assistant',
            daemon=True,
        )
        try:
            with self._lock:
                if self._closing:
                    raise RuntimeError('Assistant is shutting down')
                self._worker = worker
                worker.start()
        except BaseException:
            with self._lock:
                self._worker = None
                self._job_state.finish(profile=profile.label, ok=False)
            raise
        return self.snapshot()

    def shutdown(self) -> None:
        with self._lock:
            self._closing = True
            worker = self._worker
            ping_workers = list(self._ping_workers)
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        for ping_worker in ping_workers:
            if ping_worker is not threading.current_thread():
                ping_worker.join()

    def start_provider_login(self, params: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings()
        provider_id = self._provider_id_from_params(params, settings)
        login_fn = self._provider_operations.login
        if login_fn is None:
            return {
                'id': provider_id,
                'label': provider_id,
                'started': False,
                'message': 'Assistant service does not support provider login.',
                'errorCode': 'login_not_supported',
                'suggestion': '',
                'localHome': '',
            }
        result = login_fn(
            provider_id,
            options=self._runtime_options(settings),
            device_auth=bool(
                first_param(params, 'deviceAuth', 'device_auth', default=False)
            ),
        )
        return result.as_dict()

    def start_login(self, params: dict[str, Any]) -> dict[str, Any]:
        return self.start_provider_login(params)

    def ping_provider(self, params: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings()
        snapshot = params.get('profile')
        if snapshot is not None:
            if not isinstance(snapshot, dict):
                raise ValueError('Profile snapshot must be an object')
            profile = parse_codex_profiles([snapshot])[0]
            if (
                params.get('profileId')
                and str(params['profileId']) != profile.id
            ):
                raise ValueError('Profile snapshot id does not match profileId')
            provider_id = str(params.get('providerId') or profile.provider_id)
        else:
            profile = self._profile_from_params(params, settings)
            provider_id = self._provider_id_from_params(params, settings)
        request_id = params.get('requestId')
        request_id = '' if request_id is None else str(request_id)
        ping_fn = self._provider_operations.ping
        if ping_fn is None:
            return {
                'id': provider_id,
                'label': provider_id,
                'ok': False,
                'message': 'Assistant service does not support provider ping.',
                'errorCode': 'ping_not_supported',
                'retryable': False,
                'suggestion': '',
                'statusCode': 0,
            }
        options = self._runtime_options(settings)

        def run_ping() -> None:
            try:
                self._run_ping(
                    ping_fn,
                    provider_id,
                    options,
                    profile,
                    request_id=request_id,
                )
            finally:
                with self._lock:
                    self._ping_workers.discard(threading.current_thread())

        thread = threading.Thread(
            target=run_ping,
            daemon=True,
            name=f'assistant-ping-{provider_id}',
        )
        with self._lock:
            if self._closing:
                raise RuntimeError('Assistant is shutting down')
            self._ping_workers.add(thread)
            try:
                thread.start()
            except BaseException:
                self._ping_workers.discard(thread)
                raise
        return {'pending': True, 'providerId': provider_id}

    def _run_ping(
        self,
        ping_fn: ProviderPing,
        provider_id: str,
        options: AssistantRuntimeOptions,
        profile: CodexProfile | None,
        *,
        request_id: str = '',
    ) -> None:
        try:
            result = ping_fn(provider_id, options=options, profile=profile)
            payload = result.as_dict()
        except Exception as error:
            payload = {
                'id': provider_id,
                'ok': False,
                'message': str(error),
                'errorCode': 'ping_failed',
                'retryable': True,
            }
        self._emit(
            'assistant_ping_result',
            {
                **payload,
                'profileId': profile.id if profile is not None else '',
                'requestId': request_id,
            },
        )
        # Refresh provider cache so snapshot reflects updated status
        try:
            settings = self._settings()
            self._refresh_provider_records(settings)
        except Exception:
            _logger.debug(
                'Best-effort operation failed in _run_ping', exc_info=True
            )

    def start_local_model(self, params: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings()
        profile = self._profile_from_params(params, settings)
        if profile is None:
            raise ValueError('Profile not found')

        def _emit(payload: dict[str, Any]) -> None:
            event_type = str(payload.get('type') or 'local_llm_status')
            body = dict(payload)
            body.pop('type', None)
            self._emit(event_type, body)

        with self._lock:
            if self._closing:
                raise RuntimeError('Assistant is shutting down')
            return self.local_models.start(profile, self.project_root, _emit)

    def stop_local_model(self, params: dict[str, Any]) -> dict[str, Any]:
        settings = self._settings()
        profile = self._profile_from_params(params, settings)
        if profile is None:
            raise ValueError('Profile not found')

        return self.local_models.stop(profile)

    def _run_worker(
        self, command: InvokeAssistantCommand, settings: CodexSettings
    ) -> None:
        try:
            self.assistant_service.execute(
                command,
                options=self._runtime_options(settings),
                publish_event=self._publish_service_event,
            )
        except Exception as error:
            _logger.exception('Assistant request failed')
            text = f'{type(error).__name__}: {error}'
            event = CodexResultEvent(
                ok=False,
                profile=command.profile.label,
                cmd=command.request_text,
                text=text,
                dt_s=0.0,
                provider=command.profile.provider_id,
                model=command.profile.model,
                error_code='unknown_error',
                retryable=isinstance(error, (OSError, TimeoutError)),
            )
            self._publish_service_event(event)
        finally:
            with self._lock:
                if self._worker is threading.current_thread():
                    self._worker = None

    def _publish_service_event(self, event: object) -> None:
        record = event_to_record(event)
        if isinstance(event, CodexFallbackStartedEvent):
            with self._lock:
                if self._job_state.is_busy:
                    self._job_state.begin_fallback(
                        profile=event.profile, reason=event.reason
                    )
            self._emit('assistant_fallback', record)
            return

        if isinstance(event, CodexResultEvent):
            with self._lock:
                self._last_response = {
                    'ok': bool(event.ok),
                    'profile': str(event.profile),
                    'cmd': str(event.cmd),
                    'text': str(event.text),
                    'dtS': float(event.dt_s),
                    'provider': str(event.provider),
                    'model': str(event.model),
                    'error': {
                        'code': str(event.error_code),
                        'retryable': bool(event.retryable),
                        'suggestion': str(event.suggestion),
                        'details': str(event.details),
                    }
                    if not event.ok
                    else {},
                    'ts': float(event.ts),
                }
                if not event.ok:
                    self._last_error = str(event.text)
                if self._job_state.is_busy:
                    self._job_state.finish(
                        profile=event.profile, ok=event.ok, elapsed_s=event.dt_s
                    )
            self._emit('assistant_result', deepcopy(self._last_response))
            return

        self._emit('assistant_event', record)

    def _settings(self) -> CodexSettings:
        try:
            config = self.config_repository.read()
        except (OSError, ValueError):
            _logger.warning(
                'Unable to read assistant configuration', exc_info=True
            )
            config = {}
        codex = config.get('codex', {}) if isinstance(config, dict) else {}
        return parse_codex_settings(codex)

    def _runtime_options(
        self, settings: CodexSettings
    ) -> AssistantRuntimeOptions:
        return AssistantRuntimeOptions(
            project_root=Path(self.project_root),
            default_max_log_chars=int(settings.max_log_chars),
            answer_keyword=str(settings.answer_keyword or 'ANSWER'),
            command_tokens=list(settings.command_tokens),
            path_hints=list(settings.path_hints),
            proxy=str(settings.proxy or ''),
            default_timeout_s=int(settings.timeout_s),
            profiles=list(settings.profiles),
        )

    def _provider_id_from_params(
        self, params: dict[str, Any], settings: CodexSettings
    ) -> str:
        profile = self._profile_from_params(params, settings)
        return str(
            params.get('providerId')
            or params.get('provider_id')
            or (profile.provider_id if profile is not None else '')
            or ASSISTANT_PROVIDER_CODEX
        )

    def _profile_from_params(
        self, params: dict[str, Any], settings: CodexSettings
    ) -> CodexProfile | None:
        if not settings.profiles:
            return None
        profile_id = str(
            first_param(params, 'profileId', 'profile_id', default='') or ''
        ).strip()
        if profile_id:
            return self._select_profile(settings, profile_id, strict=True)
        return self._selected_profile(settings)

    def _provider_records(
        self, settings: CodexSettings
    ) -> list[dict[str, Any]]:
        # Return cached status only — never probe providers automatically.
        # Status is refreshed only via explicit user actions (ping, invoke).
        cache_key = provider_cache_key(settings)
        now = time.monotonic()
        with self._lock:
            if (
                self._provider_cache_key == cache_key
                and self._provider_cache_records
            ):
                if now - self._provider_cache_ts < 30.0:
                    return deepcopy(self._provider_cache_records)
        return []

    def _refresh_provider_records(
        self, settings: CodexSettings
    ) -> list[dict[str, Any]]:
        cache_key = provider_cache_key(settings)
        status_fn = self._provider_operations.statuses
        if status_fn is not None:
            try:
                records = [
                    status.as_dict()
                    for status in status_fn(
                        options=self._runtime_options(settings)
                    )
                ]
                return self._cache_provider_records(cache_key, records)
            except Exception as error:
                return self._cache_provider_records(
                    cache_key,
                    [
                        {
                            'id': ASSISTANT_PROVIDER_CODEX,
                            'label': 'Codex CLI',
                            'available': False,
                            'message': f'{type(error).__name__}: {error}',
                            'errorCode': 'provider_status_error',
                            'retryable': True,
                            'suggestion': '',
                            'models': [],
                        }
                    ],
                )
        return self._cache_provider_records(
            cache_key,
            [
                {
                    'id': ASSISTANT_PROVIDER_CODEX,
                    'label': 'Codex CLI',
                    'available': True,
                    'message': '',
                    'errorCode': '',
                    'retryable': False,
                    'suggestion': '',
                    'models': [],
                }
            ],
        )

    def _cache_provider_records(
        self, cache_key: tuple, records: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        with self._lock:
            self._provider_cache_key = cache_key
            self._provider_cache_ts = time.monotonic()
            self._provider_cache_records = deepcopy(records)
            return deepcopy(self._provider_cache_records)

    def _select_profile(
        self, settings: CodexSettings, profile_id: str, *, strict: bool = False
    ) -> CodexProfile:
        if not settings.profiles:
            raise RuntimeError('No assistant profiles are configured')
        wanted = str(profile_id or settings.selected_profile_id or '').strip()
        for profile in settings.profiles:
            if profile.id == wanted:
                return profile
        if strict and wanted:
            raise ValueError(f'Unknown assistant profile: {wanted}')
        return settings.profiles[0]

    def _selected_profile(self, settings: CodexSettings) -> CodexProfile:
        return self._select_profile(settings, settings.selected_profile_id)

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        sink = self.event_sink
        if sink is None:
            return
        try:
            sink(event_type, {'ts': time.time(), **payload})
        except Exception:
            _logger.debug(
                'Best-effort operation failed in _emit', exc_info=True
            )


def _profile_record(profile: CodexProfile) -> dict[str, Any]:
    provider_id = profile.provider_id
    return {
        'id': profile.id,
        'label': profile.label,
        'providerId': provider_id,
        'provider': provider_id,
        'model': profile.model,
        'reasoningEffort': profile.reasoning_effort,
        'baseUrl': profile.base_url,
        'offline': str(provider_id).lower() != ASSISTANT_PROVIDER_CODEX,
    }
