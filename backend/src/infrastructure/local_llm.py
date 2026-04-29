from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from application.codex_prompting import normalize_model_name
from assistant.application.provider import (
    ASSISTANT_PROVIDER_OLLAMA,
    ASSISTANT_PROVIDER_OPENAI_LOCAL,
    AssistantExecutionSettings,
    AssistantProviderError,
    AssistantProviderInfo,
    AssistantProviderPingResult,
    AssistantProviderRequest,
    AssistantProviderResult,
    normalize_provider_id,
    result_from_error,
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
        except LocalLlmError as exc:
            return _status_error(self.provider_id, self.provider_label, exc, base_url)

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
        except LocalLlmError as exc:
            return _ping_error(self.provider_id, self.provider_label, exc)
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
        except LocalLlmError as exc:
            return _error_result(profile, request.original_cmd, self.provider_id, model, exc.as_provider_error(), started)

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
        except LocalLlmError as exc:
            return _status_error(self.provider_id, self.provider_label, exc, base_url)

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
        except LocalLlmError as exc:
            return _ping_error(self.provider_id, self.provider_label, exc)
        return _ping_ok(self.provider_id, self.provider_label, f"Local endpoint is reachable at {base_url}.")

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

        try:
            data = _request_json(
                "POST",
                f"{_base_url(profile, OPENAI_LOCAL_DEFAULT_URL)}/chat/completions",
                payload=body,
                headers=_auth_header(profile),
                timeout_s=request.settings.timeout_s,
            )
        except LocalLlmError as exc:
            return _error_result(profile, request.original_cmd, self.provider_id, model, exc.as_provider_error(), started)

        text = _openai_text(data)
        return _ok_result(profile, request.original_cmd, self.provider_id, model, text or "(empty response)", started)


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


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: int,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = {"Accept": "application/json", **dict(headers or {})}
    if data is not None:
        request_headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        response = urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            request,
            timeout=max(1, int(timeout_s or 1)),
        )
    except urllib.error.HTTPError as exc:
        status = int(exc.code or 0)
        body = _http_error_body(exc)
        raise LocalLlmError(
            code="local_llm_http_error",
            message=f"Local LLM endpoint returned HTTP {status}: {body or getattr(exc, 'reason', '')}",
            retryable=status >= 500 or status == 429,
            suggestion=_http_suggestion(status),
            status_code=status,
        ) from exc
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        raise LocalLlmError(
            code="local_llm_unavailable",
            message=f"{type(exc).__name__}: {exc}",
            suggestion="Start the local LLM server and check the profile base URL.",
        ) from exc

    try:
        raw = response.read().decode("utf-8", errors="replace")
    finally:
        response.close()
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalLlmError(
            code="local_llm_bad_response",
            message=f"Local LLM endpoint returned invalid JSON: {exc}",
            suggestion="Check that the selected provider type matches the local server API.",
        ) from exc
    return parsed if isinstance(parsed, dict) else {"data": parsed}


def _profile_for_provider(settings: AssistantExecutionSettings, provider_id: str) -> Any | None:
    selected = getattr(settings, "profile", None)
    if _same_provider(selected, provider_id):
        return selected
    for profile in list(getattr(settings, "profiles", []) or []):
        if _same_provider(profile, provider_id):
            return profile
    return None


def _same_provider(profile: Any, provider_id: str) -> bool:
    if profile is None:
        return False
    return normalize_provider_id(getattr(profile, "provider_id", "")) == normalize_provider_id(provider_id)


def _base_url(profile: Any, default: str) -> str:
    raw = str(getattr(profile, "base_url", "") or "").strip().rstrip("/")
    return raw or default


def _status_timeout(settings: AssistantExecutionSettings) -> int:
    return min(3, max(1, int(getattr(settings, "timeout_s", 2) or 2)))


def _temperature(profile: Any) -> float | None:
    raw = getattr(profile, "temperature", None)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return max(0.0, min(2.0, float(str(raw).replace(",", "."))))
    except Exception:
        return None


def _max_tokens(profile: Any) -> int:
    try:
        return max(0, int(getattr(profile, "max_tokens", 0) or 0))
    except Exception:
        return 0


def _ollama_options(profile: Any) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if (temperature := _temperature(profile)) is not None:
        options["temperature"] = temperature
    if (max_tokens := _max_tokens(profile)) > 0:
        options["num_predict"] = max_tokens
    return options


def _auth_header(profile: Any) -> dict[str, str]:
    api_key = str(getattr(profile, "api_key", "") or "").strip()
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _ollama_models(data: dict[str, Any]) -> list[str]:
    return [
        str(item.get("name") or item.get("model") or "").strip()
        for item in data.get("models", [])
        if isinstance(item, dict) and str(item.get("name") or item.get("model") or "").strip()
    ]


def _openai_models(data: dict[str, Any]) -> list[str]:
    return [
        str(item.get("id") or "").strip()
        for item in data.get("data", [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    ]


def _openai_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message")
    if isinstance(message, dict):
        return str(message.get("content") or "").strip()
    return str(choices[0].get("text") or "").strip()


def _status_error(provider_id: str, label: str, error: LocalLlmError, base_url: str) -> AssistantProviderInfo:
    return AssistantProviderInfo(
        id=provider_id,
        label=label,
        available=False,
        message=f"{error.message} ({base_url})",
        error_code=error.code,
        retryable=error.retryable,
        suggestion=error.suggestion,
    )


def _ping_ok(provider_id: str, label: str, message: str) -> AssistantProviderPingResult:
    return AssistantProviderPingResult(id=provider_id, label=label, ok=True, message=message, status_code=200)


def _ping_error(provider_id: str, label: str, error: LocalLlmError) -> AssistantProviderPingResult:
    return AssistantProviderPingResult(
        id=provider_id,
        label=label,
        ok=False,
        message=error.message,
        error_code=error.code,
        retryable=error.retryable,
        suggestion=error.suggestion,
        status_code=error.status_code,
    )


def _model_required(label: str) -> AssistantProviderError:
    return AssistantProviderError(
        code="model_required",
        message=f"{label} profile requires a model name.",
        suggestion="Set the assistant profile model to one of the models loaded in the local runtime.",
    )


def _error_result(
    profile: Any,
    cmd: str,
    provider_id: str,
    model: str,
    error: AssistantProviderError,
    started: float,
) -> AssistantProviderResult:
    return result_from_error(
        profile=profile,
        cmd=cmd,
        provider=provider_id,
        model=model,
        error=error,
        started_at=started,
    )


def _ok_result(
    profile: Any,
    cmd: str,
    provider_id: str,
    model: str,
    text: str,
    started: float,
) -> AssistantProviderResult:
    return AssistantProviderResult(
        ok=True,
        profile=str(getattr(profile, "label", "") or ""),
        cmd=str(cmd),
        text=text,
        dt_s=max(0.0, time.time() - started),
        provider=provider_id,
        model=model,
    )


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def _http_suggestion(status: int) -> str:
    if status == 404:
        return "Check the provider type, base URL, and model name."
    if status in (401, 403):
        return "Check the local endpoint API key or disable auth in the local server."
    if status == 429:
        return "The local server is busy; retry or use a smaller model."
    if status >= 500:
        return "Check the local LLM server logs."
    return "Check the selected local LLM profile."
