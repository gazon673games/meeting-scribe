from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
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
            message = _ensure_openai_local_runtime(profile, settings, base_url)
        except LocalLlmError as exc:
            return _ping_error(self.provider_id, self.provider_label, exc)
        return _ping_ok(self.provider_id, self.provider_label, message)

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
        except LocalLlmError as exc:
            return _error_result(profile, request.original_cmd, self.provider_id, model, exc.as_provider_error(), started)

        try:
            data = _request_json(
                "POST",
                f"{base_url}/chat/completions",
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
        except LocalLlmError as exc:
            _emit("error", exc.message)
        except Exception as exc:
            _emit("error", str(exc))

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


def _ensure_openai_local_runtime(profile: Any, settings: AssistantExecutionSettings, base_url: str) -> str:
    try:
        _request_json("GET", f"{base_url}/models", timeout_s=_status_timeout(settings))
        return f"Local endpoint is reachable at {base_url}."
    except LocalLlmError as exc:
        if exc.code != "local_llm_unavailable":
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

    with _SERVER_LOCK:
        process = _SERVER_PROCESSES.get(key)
        if process is None or process.poll() is not None:
            _SERVER_PROCESSES[key] = _start_llama_server(server, model_path, model, host, port)

    deadline = time.time() + min(45, max(10, int(getattr(settings, "timeout_s", 30) or 30)))
    last_error: LocalLlmError | None = None
    while time.time() < deadline:
        try:
            _request_json("GET", f"{base_url}/models", timeout_s=1)
            return f"Started local llama.cpp server at {base_url}."
        except LocalLlmError as exc:
            last_error = exc
            with _SERVER_LOCK:
                process = _SERVER_PROCESSES.get(key)
            if process is not None and process.poll() is not None:
                raise LocalLlmError(
                    code="local_llm_server_exited",
                    message=f"llama-server exited with code {process.returncode}",
                    suggestion="Check the model file, GPU memory, and llama.cpp runtime files.",
                ) from exc
            time.sleep(0.5)

    raise LocalLlmError(
        code="local_llm_start_timeout",
        message=f"llama-server did not become ready at {base_url}",
        suggestion=(last_error.suggestion if last_error else "Check the local LLM server logs."),
    )


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


def _find_llama_server(project_root: Path) -> Path:
    env_path = str(os.environ.get("LLAMA_SERVER_EXE", "")).strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(sorted((project_root / ".local" / "llama_cpp").glob("**/llama-server.exe")))
    candidates.extend(sorted((project_root / ".local" / "llama_cpp").glob("**/llama-server")))
    if found := shutil.which("llama-server"):
        candidates.append(Path(found))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise LocalLlmError(
        code="local_llm_server_missing",
        message="llama-server executable was not found",
        suggestion="Install llama.cpp locally or set LLAMA_SERVER_EXE to llama-server.",
    )


def _find_gguf_model(project_root: Path, model: str) -> Path:
    text = str(model or "").strip()
    if not text:
        raise LocalLlmError(
            code="model_required",
            message="Local OpenAI-compatible profile requires a model name.",
            suggestion="Select a GGUF model in Settings > Models > Language Models.",
        )

    direct = Path(text).expanduser()
    direct_candidates = [direct]
    if not direct.is_absolute():
        direct_candidates.append(project_root / direct)
    for candidate in direct_candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".gguf":
            return candidate.resolve()

    wanted = Path(text).stem if text.lower().endswith(".gguf") else text
    models_root = project_root / "models" / "llm"
    matches = [
        path
        for path in sorted(models_root.rglob("*.gguf"), key=lambda item: str(item).lower())
        if path.stem == wanted or path.name == text
    ]
    if matches:
        return matches[0].resolve()

    raise LocalLlmError(
        code="local_llm_model_missing",
        message=f"GGUF model '{text}' was not found",
        suggestion="Download or choose a GGUF model from Settings > Models > Language Models.",
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
