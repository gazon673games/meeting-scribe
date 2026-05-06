from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from application.codex_prompting import normalize_model_name
from assistant.application.provider import (
    ASSISTANT_PROVIDER_OLLAMA,
    ASSISTANT_PROVIDER_OPENAI_LOCAL,
    AssistantExecutionSettings,
    AssistantProviderInfo,
    AssistantProviderPingResult,
    AssistantProviderRequest,
    AssistantProviderResult,
)
from infrastructure.local_llm_parts import (
    LocalLlmError,
    auth_header as _auth_header,
    base_url as _base_url,
    ensure_openai_local_runtime as _ensure_openai_local_runtime,
    error_result as _error_result,
    max_tokens as _max_tokens,
    model_required as _model_required,
    ok_result as _ok_result,
    ollama_models as _ollama_models,
    ollama_options as _ollama_options,
    openai_models as _openai_models,
    openai_text as _openai_text,
    ping_error as _ping_error,
    ping_ok as _ping_ok,
    profile_for_provider as _profile_for_provider,
    request_json as _request_json,
    start_local_llm_async as _start_local_llm_async,
    status_error as _status_error,
    status_timeout as _status_timeout,
    stop_local_llm as _stop_local_llm,
    temperature as _temperature,
    SERVER_PROCESSES as _SERVER_PROCESSES,
)


OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
OPENAI_LOCAL_DEFAULT_URL = "http://127.0.0.1:1234/v1"


class OllamaLocalLlmRunner:
    provider_id = ASSISTANT_PROVIDER_OLLAMA
    provider_label = "Ollama"

    def status(self, settings: AssistantExecutionSettings) -> AssistantProviderInfo:
        profile = _profile_for_provider(settings, self.provider_id)
        base_url = _base_url(profile, OLLAMA_DEFAULT_URL)
        try:
            data = _request_json("GET", f"{base_url}/api/tags", timeout_s=_status_timeout(settings))
        except LocalLlmError as error:
            return _status_error(self.provider_id, self.provider_label, error, base_url)

        models = _ollama_models(data)
        return AssistantProviderInfo(
            id=self.provider_id,
            label=self.provider_label,
            available=True,
            message=f"Ollama is reachable at {base_url}.",
            suggestion="" if models else "Pull a model with ollama pull <model>.",
            models=models,
        )

    def ping(self, settings: AssistantExecutionSettings) -> AssistantProviderPingResult:
        profile = _profile_for_provider(settings, self.provider_id)
        base_url = _base_url(profile, OLLAMA_DEFAULT_URL)
        try:
            _request_json("GET", f"{base_url}/api/tags", timeout_s=_status_timeout(settings))
        except LocalLlmError as error:
            return _ping_error(self.provider_id, self.provider_label, error)
        return _ping_ok(self.provider_id, self.provider_label, f"Ollama is reachable at {base_url}.")

    def run(self, request: AssistantProviderRequest) -> AssistantProviderResult:
        started = time.time()
        profile = request.profile
        model = normalize_model_name(getattr(profile, "model", ""))
        if not model:
            return _error_result(profile, request.original_cmd, self.provider_id, model, _model_required("Ollama"), started)

        body: dict[str, Any] = {"model": model, "prompt": request.prompt, "stream": False}
        options = _ollama_options(profile)
        if options:
            body["options"] = options

        try:
            data = _request_json(
                "POST",
                f"{_base_url(profile, OLLAMA_DEFAULT_URL)}/api/generate",
                payload=body,
                timeout_s=request.settings.timeout_s,
            )
        except LocalLlmError as error:
            return _error_result(
                profile,
                request.original_cmd,
                self.provider_id,
                model,
                error.as_provider_error(),
                started,
            )

        text = str(data.get("response") or "").strip()
        return _ok_result(profile, request.original_cmd, self.provider_id, model, text or "(empty response)", started)


class OpenAICompatibleLocalLlmRunner:
    provider_id = ASSISTANT_PROVIDER_OPENAI_LOCAL
    provider_label = "Local OpenAI-compatible"

    def status(self, settings: AssistantExecutionSettings) -> AssistantProviderInfo:
        profile = _profile_for_provider(settings, self.provider_id)
        base_url = _base_url(profile, OPENAI_LOCAL_DEFAULT_URL)
        try:
            data = _request_json("GET", f"{base_url}/models", timeout_s=_status_timeout(settings))
        except LocalLlmError as error:
            return _status_error(self.provider_id, self.provider_label, error, base_url)

        models = _openai_models(data)
        return AssistantProviderInfo(
            id=self.provider_id,
            label=self.provider_label,
            available=True,
            message=f"OpenAI-compatible local endpoint is reachable at {base_url}.",
            suggestion="" if models else "Load a model in the local OpenAI-compatible runtime.",
            models=models,
        )

    def ping(self, settings: AssistantExecutionSettings) -> AssistantProviderPingResult:
        profile = _profile_for_provider(settings, self.provider_id)
        base_url = _base_url(profile, OPENAI_LOCAL_DEFAULT_URL)
        try:
            _request_json("GET", f"{base_url}/models", timeout_s=_status_timeout(settings))
        except LocalLlmError as error:
            return _ping_error(self.provider_id, self.provider_label, error)
        return _ping_ok(self.provider_id, self.provider_label, f"OpenAI-compatible endpoint is reachable at {base_url}.")

    def run(self, request: AssistantProviderRequest) -> AssistantProviderResult:
        started = time.time()
        profile = request.profile
        model = normalize_model_name(getattr(profile, "model", ""))
        if not model:
            return _error_result(
                profile,
                request.original_cmd,
                self.provider_id,
                model,
                _model_required("OpenAI-compatible local"),
                started,
            )

        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
        }
        if (temperature := _temperature(profile)) is not None:
            body["temperature"] = temperature
        if (max_tokens := _max_tokens(profile)) > 0:
            body["max_tokens"] = max_tokens

        base_url = _base_url(profile, OPENAI_LOCAL_DEFAULT_URL)
        try:
            _ensure_openai_local_runtime(profile, request.settings, base_url)
        except LocalLlmError as error:
            return _error_result(
                profile,
                request.original_cmd,
                self.provider_id,
                model,
                error.as_provider_error(),
                started,
            )

        try:
            data = _request_json(
                "POST",
                f"{base_url}/chat/completions",
                payload=body,
                headers=_auth_header(profile),
                timeout_s=request.settings.timeout_s,
            )
        except LocalLlmError as error:
            return _error_result(
                profile,
                request.original_cmd,
                self.provider_id,
                model,
                error.as_provider_error(),
                started,
            )

        text = _openai_text(data)
        return _ok_result(profile, request.original_cmd, self.provider_id, model, text or "(empty response)", started)


def start_local_llm_async(profile: Any, project_root: Path, emit_event: Any) -> dict:
    return _start_local_llm_async(
        profile=profile,
        project_root=project_root,
        emit_event=emit_event,
        default_base_url=OPENAI_LOCAL_DEFAULT_URL,
    )


def stop_local_llm(profile: Any) -> dict:
    return _stop_local_llm(profile=profile, default_base_url=OPENAI_LOCAL_DEFAULT_URL)


__all__ = [
    "LocalLlmError",
    "OLLAMA_DEFAULT_URL",
    "OPENAI_LOCAL_DEFAULT_URL",
    "OllamaLocalLlmRunner",
    "OpenAICompatibleLocalLlmRunner",
    "_SERVER_PROCESSES",
    "_request_json",
    "start_local_llm_async",
    "stop_local_llm",
]
