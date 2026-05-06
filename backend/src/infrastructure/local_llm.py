from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
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
    error_result as _error_result,
    find_direct_gguf_path as _find_direct_gguf_path,
    find_gguf_model as _find_gguf_model,
    find_llama_server as _find_llama_server,
    http_error_body as _http_error_body,
    http_suggestion as _http_suggestion,
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
    status_error as _status_error,
    status_timeout as _status_timeout,
    temperature as _temperature,
)


OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
OPENAI_LOCAL_DEFAULT_URL = "http://127.0.0.1:1234/v1"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
_SERVER_LOCK = threading.RLock()
_SERVER_PROCESSES: dict[str, subprocess.Popen] = {}


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
    """Start llama-server for an openai_local profile in a background thread."""
    profile_id = str(getattr(profile, "id", "") or getattr(profile, "label", "") or "local")
    base_url = _base_url(profile, OPENAI_LOCAL_DEFAULT_URL)

    class _Settings:
        timeout_s = 60
        project_root = None

    _Settings.project_root = project_root

    def _run() -> None:
        def _emit(state: str, message: str = "") -> None:
            if callable(emit_event):
                emit_event({"type": "local_llm_status", "profileId": profile_id, "state": state, "message": message})

        try:
            _emit("starting")
            msg = _ensure_openai_local_runtime(profile, _Settings(), base_url)
            _emit("running", msg)
        except LocalLlmError as error:
            _emit("error", error.message)
        except Exception as error:  # pragma: no cover - safety net
            _emit("error", str(error))

    threading.Thread(target=_run, daemon=True, name=f"llm-start-{profile_id}").start()
    return {"started": True, "profileId": profile_id}


def stop_local_llm(profile: Any) -> dict:
    """Terminate llama-server started for a given profile's base_url host:port."""
    profile_id = str(getattr(profile, "id", "") or getattr(profile, "label", "") or "local")
    base_url = _base_url(profile, OPENAI_LOCAL_DEFAULT_URL)
    parsed = urllib.parse.urlparse(base_url)
    host = (parsed.hostname or "127.0.0.1").lower()
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    prefix = f"{host}:{port}:"

    killed = 0
    with _SERVER_LOCK:
        for key in list(_SERVER_PROCESSES.keys()):
            if key.startswith(prefix):
                process = _SERVER_PROCESSES.pop(key)
                if process.poll() is None:
                    process.terminate()
                    killed += 1

    return {"stopped": True, "profileId": profile_id, "killed": killed}


def _start_server_if_needed(key: str, server: Path, model_path: Path, model: str, host: str, port: int) -> None:
    with _SERVER_LOCK:
        dead_keys = [item_key for item_key, process in list(_SERVER_PROCESSES.items()) if process.poll() is not None]
        for dead_key in dead_keys:
            del _SERVER_PROCESSES[dead_key]
        process = _SERVER_PROCESSES.get(key)
        if process is None or process.poll() is not None:
            _SERVER_PROCESSES[key] = _start_llama_server(server, model_path, model, host, port)


def _poll_until_server_ready(base_url: str, settings: AssistantExecutionSettings, key: str) -> str:
    deadline = time.time() + min(45, max(10, int(getattr(settings, "timeout_s", 30) or 30)))
    last_error: LocalLlmError | None = None
    while time.time() < deadline:
        try:
            _request_json("GET", f"{base_url}/models", timeout_s=1)
            return f"Started local llama.cpp server at {base_url}."
        except LocalLlmError as error:
            last_error = error
            with _SERVER_LOCK:
                process = _SERVER_PROCESSES.get(key)
            if process is not None and process.poll() is not None:
                raise LocalLlmError(
                    code="local_llm_server_exited",
                    message=f"llama-server exited with code {process.returncode}",
                    suggestion="Check the model file, GPU memory, and llama.cpp runtime files.",
                ) from error
            time.sleep(0.5)
    raise LocalLlmError(
        code="local_llm_start_timeout",
        message=f"llama-server did not become ready at {base_url}",
        suggestion=(last_error.suggestion if last_error else "Check the local LLM server logs."),
    )


def _ensure_openai_local_runtime(profile: Any, settings: AssistantExecutionSettings, base_url: str) -> str:
    try:
        _request_json("GET", f"{base_url}/models", timeout_s=_status_timeout(settings))
        return f"Local endpoint is reachable at {base_url}."
    except LocalLlmError as error:
        if error.code != "local_llm_unavailable":
            raise

    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    if host.lower() not in LOCAL_HOSTS:
        raise LocalLlmError(
            code="local_llm_unavailable",
            message=f"Local LLM endpoint is not reachable at {base_url}",
            suggestion="Start the remote/local LLM server or choose a localhost profile.",
        )

    model = normalize_model_name(getattr(profile, "model", ""))
    model_path = _find_gguf_model(Path(settings.project_root or "."), model)
    server = _find_llama_server(Path(settings.project_root or "."))
    key = f"{host}:{port}:{model_path}"
    _start_server_if_needed(key, server, model_path, model, host, port)
    return _poll_until_server_ready(base_url, settings, key)


def _start_llama_server(server: Path, model_path: Path, alias: str, host: str, port: int) -> subprocess.Popen:
    command = [
        str(server),
        "-m",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "-a",
        alias,
        "-c",
        os.environ.get("LLAMA_SERVER_CTX", "4096"),
    ]
    ngl = str(os.environ.get("LLAMA_SERVER_NGL", "28")).strip()
    if ngl:
        command.extend(["-ngl", ngl])

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(
        command,
        cwd=str(server.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def _http_open(request: urllib.request.Request, timeout_s: int) -> Any:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=max(1, int(timeout_s or 1)))
    except urllib.error.HTTPError as error:
        status = int(error.code or 0)
        body = _http_error_body(error)
        raise LocalLlmError(
            code="local_llm_http_error",
            message=f"Local LLM endpoint returned HTTP {status}: {body or getattr(error, 'reason', '')}",
            retryable=status >= 500 or status == 429,
            suggestion=_http_suggestion(status),
            status_code=status,
        ) from error
    except (TimeoutError, urllib.error.URLError, OSError) as error:
        raise LocalLlmError(
            code="local_llm_unavailable",
            message=f"{type(error).__name__}: {error}",
            suggestion="Start the local LLM server and check the profile base URL.",
        ) from error


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

    response = _http_open(
        urllib.request.Request(url, data=data, headers=request_headers, method=method),
        timeout_s,
    )
    try:
        raw = response.read().decode("utf-8", errors="replace")
    finally:
        response.close()
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LocalLlmError(
            code="local_llm_bad_response",
            message=f"Local LLM endpoint returned invalid JSON: {error}",
            suggestion="Check that the selected provider type matches the local server API.",
        ) from error
    return parsed if isinstance(parsed, dict) else {"data": parsed}


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
