from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from application.codex_config import CodexProfile, CodexSettings, parse_codex_settings
from application.commands import InvokeAssistantCommand
from application.event_types import CodexFallbackStartedEvent, CodexResultEvent, event_to_record
from assistant.application.service import AssistantApplicationService, AssistantRuntimeOptions
from assistant.application.provider import ASSISTANT_PROVIDER_CODEX
from assistant.domain.aggregate import AssistantJobAggregate
from settings.application.config_repository import ConfigRepository

from interface.session_controller import HeadlessSessionController

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


@dataclass
class AssistantController:
    project_root: Path
    config_repository: ConfigRepository
    assistant_service: AssistantApplicationService
    session_controller: Optional[HeadlessSessionController] = None
    event_sink: Any = None
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _job_state: AssistantJobAggregate = field(default_factory=AssistantJobAggregate)
    _last_response: Dict[str, Any] = field(default_factory=dict)
    _last_error: str = ""
    _last_request: Dict[str, Any] = field(default_factory=dict)
    _provider_cache_key: tuple = field(default_factory=tuple, init=False, repr=False)
    _provider_cache_ts: float = field(default=0.0, init=False, repr=False)
    _provider_cache_records: list[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    def set_event_sink(self, event_sink) -> None:  # noqa: ANN001
        with self._lock:
            self.event_sink = event_sink

    def snapshot(self) -> Dict[str, Any]:
        settings = self._settings()
        selected = self._selected_profile(settings) if settings.profiles else None
        providers = self._provider_records(settings)
        selected_provider = _provider_for_profile(providers, selected) if selected is not None else None
        with self._lock:
            return {
                "enabled": bool(settings.enabled),
                "providerAvailable": bool(selected_provider.get("available", False)) if selected_provider else False,
                "providerId": str(selected_provider.get("id", "")) if selected_provider else "",
                "providerMessage": str(selected_provider.get("message", "")) if selected_provider else "",
                "providerErrorCode": str(selected_provider.get("errorCode", "")) if selected_provider else "",
                "providerSuggestion": str(selected_provider.get("suggestion", "")) if selected_provider else "",
                "providerAuthRequired": bool(selected_provider.get("authRequired", False)) if selected_provider else False,
                "providerLoginSupported": bool(selected_provider.get("loginSupported", False)) if selected_provider else False,
                "providerLocalHome": str(selected_provider.get("localHome", "")) if selected_provider else "",
                "busy": bool(self._job_state.is_busy),
                "fallback": bool(self._job_state.is_fallback),
                "selectedProfileId": selected.id if selected is not None else "",
                "profiles": [_profile_record(profile) for profile in settings.profiles],
                "providers": providers,
                "lastResponse": dict(self._last_response),
                "lastError": self._last_error,
                "lastRequest": dict(self._last_request),
            }

    def invoke(self, params: Dict[str, Any]) -> Dict[str, Any]:
        settings = self._settings()
        if not settings.enabled:
            raise RuntimeError("Assistant is disabled in config")
        profile = self._select_profile(settings, str(params.get("profileId", params.get("profile_id", "")) or ""))
        request_text, source_label, limits = self._request_plan(params, settings)
        context_text = self._context_text(params, settings)
        context_label = "current transcript" if context_text is not None else "latest human log"
        if context_text is not None and not context_text.strip():
            raise RuntimeError("Assistant context is empty. Start transcription and wait for transcript text before asking.")

        command = InvokeAssistantCommand(
            profile=profile,
            request_text=request_text,
            source_label=source_label,
            context_source=str(settings.context_source or "transcript"),
            context_label=context_label,
            context_text=context_text,
            max_log_chars=limits.get("max_log_chars"),
            timeout_s=limits.get("timeout_s"),
            fallback_max_log_chars=limits.get("fallback_max_log_chars"),
            fallback_timeout_s=limits.get("fallback_timeout_s"),
        )

        with self._lock:
            if self._job_state.is_busy:
                raise RuntimeError("Assistant is already running")
            self._job_state.begin(profile=profile.label, source_label=source_label)
            self._last_error = ""
            self._last_request = {
                "profile": profile.label,
                "profileId": profile.id,
                "sourceLabel": source_label,
                "requestText": request_text,
                "contextLabel": context_label,
                "ts": time.time(),
            }

        self._emit("assistant_started", dict(self._last_request))
        worker = threading.Thread(target=self._run_worker, args=(command, settings), name="electron-assistant", daemon=True)
        worker.start()
        return self.snapshot()

    def start_provider_login(self, params: Dict[str, Any]) -> Dict[str, Any]:
        settings = self._settings()
        provider_id = self._provider_id_from_params(params, settings)
        login_fn = getattr(self.assistant_service, "start_provider_login", None)
        if not callable(login_fn):
            return {
                "id": provider_id,
                "label": provider_id,
                "started": False,
                "message": "Assistant service does not support provider login.",
                "errorCode": "login_not_supported",
                "suggestion": "",
                "localHome": "",
            }
        result = login_fn(
            provider_id,
            options=self._runtime_options(settings),
            device_auth=bool(params.get("deviceAuth", params.get("device_auth", False))),
        )
        return result.as_dict()

    def ping_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        settings = self._settings()
        profile = self._profile_from_params(params, settings)
        provider_id = self._provider_id_from_params(params, settings)
        ping_fn = getattr(self.assistant_service, "ping_provider", None)
        if not callable(ping_fn):
            return {
                "id": provider_id,
                "label": provider_id,
                "ok": False,
                "message": "Assistant service does not support provider ping.",
                "errorCode": "ping_not_supported",
                "retryable": False,
                "suggestion": "",
                "statusCode": 0,
            }
        options = self._runtime_options(settings)
        thread = threading.Thread(
            target=self._run_ping,
            args=(ping_fn, provider_id, options, profile),
            daemon=True,
        )
        thread.start()
        return {"pending": True, "providerId": provider_id}

    def _run_ping(self, ping_fn: Any, provider_id: str, options: Any, profile: Any) -> None:
        try:
            result = ping_fn(provider_id, options=options, profile=profile)
            payload = result.as_dict()
        except Exception as exc:
            payload = {
                "id": provider_id,
                "ok": False,
                "message": str(exc),
                "errorCode": "ping_failed",
                "retryable": True,
            }
        self._emit("assistant_ping_result", payload)

    def start_local_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from infrastructure.local_llm import start_local_llm_async
        settings = self._settings()
        profile = self._profile_from_params(params, settings)
        if profile is None:
            raise ValueError("Profile not found")

        def _emit(payload: Dict[str, Any]) -> None:
            event_type = str(payload.get("type") or "local_llm_status")
            body = dict(payload)
            body.pop("type", None)
            self._emit(event_type, body)

        return start_local_llm_async(profile, self.project_root, _emit)

    def stop_local_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from infrastructure.local_llm import stop_local_llm
        settings = self._settings()
        profile = self._profile_from_params(params, settings)
        if profile is None:
            raise ValueError("Profile not found")
        return stop_local_llm(profile)

    def _run_worker(self, command: InvokeAssistantCommand, settings: CodexSettings) -> None:
        try:
            self.assistant_service.execute(
                command,
                options=self._runtime_options(settings),
                publish_event=self._publish_service_event,
            )
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            event = CodexResultEvent(
                ok=False,
                profile=command.profile.label,
                cmd=command.request_text,
                text=text,
                dt_s=0.0,
                provider=getattr(command.profile, "provider_id", ASSISTANT_PROVIDER_CODEX),
                model=getattr(command.profile, "model", ""),
                error_code="unknown_error",
                retryable=True,
            )
            self._publish_service_event(event)

    def _publish_service_event(self, event: object) -> None:
        record = event_to_record(event)
        if isinstance(event, CodexFallbackStartedEvent):
            with self._lock:
                if self._job_state.is_busy:
                    self._job_state.begin_fallback(profile=event.profile, reason=event.reason)
            self._emit("assistant_fallback", record)
            return

        if isinstance(event, CodexResultEvent):
            with self._lock:
                self._last_response = {
                    "ok": bool(event.ok),
                    "profile": str(event.profile),
                    "cmd": str(event.cmd),
                    "text": str(event.text),
                    "dtS": float(event.dt_s),
                    "provider": str(getattr(event, "provider", ASSISTANT_PROVIDER_CODEX)),
                    "model": str(getattr(event, "model", "")),
                    "error": {
                        "code": str(getattr(event, "error_code", "")),
                        "retryable": bool(getattr(event, "retryable", False)),
                        "suggestion": str(getattr(event, "suggestion", "")),
                        "details": str(getattr(event, "details", "")),
                    } if not event.ok else {},
                    "ts": float(event.ts),
                }
                if not event.ok:
                    self._last_error = str(event.text)
                if self._job_state.is_busy:
                    self._job_state.finish(profile=event.profile, ok=event.ok, elapsed_s=event.dt_s)
            self._emit("assistant_result", self._last_response)
            return

        self._emit("assistant_event", record)

    def _settings(self) -> CodexSettings:
        try:
            config = self.config_repository.read()
        except Exception:
            config = {}
        codex = config.get("codex", {}) if isinstance(config, dict) else {}
        return parse_codex_settings(codex)

    def _runtime_options(self, settings: CodexSettings) -> AssistantRuntimeOptions:
        return AssistantRuntimeOptions(
            project_root=Path(self.project_root),
            default_max_log_chars=int(settings.max_log_chars),
            answer_keyword=str(settings.answer_keyword or "ANSWER"),
            command_tokens=list(settings.command_tokens),
            path_hints=list(settings.path_hints),
            proxy=str(settings.proxy or ""),
            default_timeout_s=int(settings.timeout_s),
            profiles=list(settings.profiles),
        )

    def _provider_id_from_params(self, params: Dict[str, Any], settings: CodexSettings) -> str:
        profile = self._profile_from_params(params, settings)
        return str(
            params.get("providerId")
            or params.get("provider_id")
            or getattr(profile, "provider_id", "")
            or ASSISTANT_PROVIDER_CODEX
        )

    def _profile_from_params(self, params: Dict[str, Any], settings: CodexSettings) -> CodexProfile | None:
        if not settings.profiles:
            return None
        profile_id = str(params.get("profileId", params.get("profile_id", "")) or "").strip()
        if profile_id:
            return self._select_profile(settings, profile_id)
        return self._selected_profile(settings)

    def _provider_records(self, settings: CodexSettings) -> list[Dict[str, Any]]:
        cache_key = _provider_cache_key(settings)
        now = time.monotonic()
        with self._lock:
            if (
                self._provider_cache_key == cache_key
                and self._provider_cache_records
                and now - self._provider_cache_ts < 5.0
            ):
                return [dict(record) for record in self._provider_cache_records]

        status_fn = getattr(self.assistant_service, "provider_statuses", None)
        if callable(status_fn):
            try:
                records = [status.as_dict() for status in status_fn(options=self._runtime_options(settings))]
                return self._cache_provider_records(cache_key, records)
            except Exception as exc:
                return self._cache_provider_records(cache_key, [
                    {
                        "id": ASSISTANT_PROVIDER_CODEX,
                        "label": "Codex CLI",
                        "available": False,
                        "message": f"{type(exc).__name__}: {exc}",
                        "errorCode": "provider_status_error",
                        "retryable": True,
                        "suggestion": "",
                        "models": [],
                    }
                ])
        return self._cache_provider_records(cache_key, [
            {
                "id": ASSISTANT_PROVIDER_CODEX,
                "label": "Codex CLI",
                "available": True,
                "message": "",
                "errorCode": "",
                "retryable": False,
                "suggestion": "",
                "models": [],
            }
        ])

    def _cache_provider_records(self, cache_key: tuple, records: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        with self._lock:
            self._provider_cache_key = cache_key
            self._provider_cache_ts = time.monotonic()
            self._provider_cache_records = [dict(record) for record in records]
            return [dict(record) for record in self._provider_cache_records]

    def _select_profile(self, settings: CodexSettings, profile_id: str) -> CodexProfile:
        if not settings.profiles:
            raise RuntimeError("No assistant profiles are configured")
        wanted = str(profile_id or settings.selected_profile_id or "").strip()
        for profile in settings.profiles:
            if profile.id == wanted:
                return profile
        return settings.profiles[0]

    def _selected_profile(self, settings: CodexSettings) -> CodexProfile:
        return self._select_profile(settings, settings.selected_profile_id)

    def _request_plan(self, params: Dict[str, Any], settings: CodexSettings) -> tuple[str, str, Dict[str, Optional[int]]]:
        action = str(params.get("action", "") or "").strip().lower()
        raw_text = str(params.get("requestText", params.get("request_text", "")) or "").strip()
        if action == "answer":
            return (
                str(settings.answer_keyword or "ANSWER"),
                "answer",
                {
                    "max_log_chars": FAST_ANSWER_LOG_CHARS,
                    "timeout_s": FAST_ANSWER_TIMEOUT_S,
                    "fallback_max_log_chars": FAST_ANSWER_FALLBACK_LOG_CHARS,
                    "fallback_timeout_s": FAST_ANSWER_FALLBACK_TIMEOUT_S,
                },
            )
        if action == "summary":
            return (
                SUMMARY_REQUEST,
                "summary",
                {
                    "max_log_chars": SUMMARY_LOG_CHARS,
                    "timeout_s": SUMMARY_TIMEOUT_S,
                    "fallback_max_log_chars": None,
                    "fallback_timeout_s": None,
                },
            )
        if action == "action_items":
            return (
                ACTION_ITEMS_REQUEST,
                "action items",
                {
                    "max_log_chars": SUMMARY_LOG_CHARS,
                    "timeout_s": SUMMARY_TIMEOUT_S,
                    "fallback_max_log_chars": None,
                    "fallback_timeout_s": None,
                },
            )
        if action == "risk_check":
            return (
                RISK_CHECK_REQUEST,
                "risk check",
                {
                    "max_log_chars": SUMMARY_LOG_CHARS,
                    "timeout_s": SUMMARY_TIMEOUT_S,
                    "fallback_max_log_chars": None,
                    "fallback_timeout_s": None,
                },
            )
        if not raw_text:
            raise ValueError("Assistant request text is empty")
        return (
            raw_text,
            str(params.get("sourceLabel", params.get("source_label", "you")) or "you"),
            {
                "max_log_chars": _optional_int(params.get("maxLogChars", params.get("max_log_chars"))),
                "timeout_s": _optional_int(params.get("timeoutS", params.get("timeout_s"))),
                "fallback_max_log_chars": _optional_int(
                    params.get("fallbackMaxLogChars", params.get("fallback_max_log_chars"))
                ),
                "fallback_timeout_s": _optional_int(params.get("fallbackTimeoutS", params.get("fallback_timeout_s"))),
            },
        )

    def _context_text(self, params: Dict[str, Any], settings: CodexSettings) -> Optional[str]:
        if "contextText" in params:
            return str(params.get("contextText") or "")
        if str(settings.context_source or "transcript") != "transcript":
            return None
        if self.session_controller is None:
            return ""
        return self.session_controller.transcript_text()

    def _emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        sink = self.event_sink
        if sink is None:
            return
        try:
            sink(event_type, {"ts": time.time(), **payload})
        except Exception:
            pass


def _profile_record(profile: CodexProfile) -> Dict[str, Any]:
    provider_id = getattr(profile, "provider_id", ASSISTANT_PROVIDER_CODEX)
    return {
        "id": profile.id,
        "label": profile.label,
        "providerId": provider_id,
        "provider": provider_id,
        "model": profile.model,
        "reasoningEffort": profile.reasoning_effort,
        "baseUrl": getattr(profile, "base_url", ""),
        "offline": str(provider_id).lower() != ASSISTANT_PROVIDER_CODEX,
    }


def _provider_for_profile(providers: list[Dict[str, Any]], profile: CodexProfile | None) -> Dict[str, Any]:
    wanted = str(getattr(profile, "provider_id", "") or ASSISTANT_PROVIDER_CODEX).strip().lower()
    for provider in providers:
        if str(provider.get("id", "")).strip().lower() == wanted:
            return provider
    return {
        "id": wanted,
        "label": wanted,
        "available": False,
        "message": f"Assistant provider '{wanted}' is not configured.",
        "errorCode": "provider_unavailable",
    }


def _provider_cache_key(settings: CodexSettings) -> tuple:
    return (
        bool(settings.enabled),
        str(settings.proxy or ""),
        tuple(settings.command_tokens or []),
        tuple(settings.path_hints or []),
        tuple(
            (
                str(profile.id),
                str(profile.provider_id),
                str(profile.model),
                str(profile.base_url),
                str(profile.codex_profile),
            )
            for profile in list(settings.profiles or [])
        ),
    )


def _optional_int(raw: object) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None
