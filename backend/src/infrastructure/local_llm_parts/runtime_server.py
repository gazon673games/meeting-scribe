from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

from application.codex_prompting import normalize_model_name
from assistant.application.provider import AssistantExecutionSettings
from infrastructure.local_llm_parts.errors import LocalLlmError
from infrastructure.local_llm_parts.http_client import request_json
from infrastructure.local_llm_parts.profile_utils import base_url
from infrastructure.local_llm_parts.runtime_discovery import find_gguf_model, find_llama_server

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SERVER_LOCK = threading.RLock()
SERVER_PROCESSES: dict[str, subprocess.Popen] = {}


def start_local_llm_async(
    profile: Any,
    project_root: Path,
    emit_event: Any,  # noqa: ANN401
    *,
    default_base_url: str,
) -> dict:
    """Start llama-server for an openai_local profile in a background thread."""
    profile_id = str(getattr(profile, "id", "") or getattr(profile, "label", "") or "local")
    resolved_base_url = base_url(profile, default_base_url)

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
            message = ensure_openai_local_runtime(profile, _Settings(), resolved_base_url)
            _emit("running", message)
        except LocalLlmError as error:
            _emit("error", error.message)
        except Exception as error:  # pragma: no cover - safety net
            _emit("error", str(error))

    threading.Thread(target=_run, daemon=True, name=f"llm-start-{profile_id}").start()
    return {"started": True, "profileId": profile_id}


def stop_local_llm(profile: Any, *, default_base_url: str) -> dict:
    """Terminate llama-server started for a given profile base_url host:port."""
    profile_id = str(getattr(profile, "id", "") or getattr(profile, "label", "") or "local")
    resolved_base_url = base_url(profile, default_base_url)
    parsed = urllib.parse.urlparse(resolved_base_url)
    host = (parsed.hostname or "127.0.0.1").lower()
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    prefix = f"{host}:{port}:"

    killed = 0
    with SERVER_LOCK:
        for key in list(SERVER_PROCESSES.keys()):
            if key.startswith(prefix):
                process = SERVER_PROCESSES.pop(key)
                if process.poll() is None:
                    process.terminate()
                    killed += 1

    return {"stopped": True, "profileId": profile_id, "killed": killed}


def ensure_openai_local_runtime(profile: Any, settings: AssistantExecutionSettings, resolved_base_url: str) -> str:
    try:
        request_json("GET", f"{resolved_base_url}/models", timeout_s=_status_timeout(settings))
        return f"Local endpoint is reachable at {resolved_base_url}."
    except LocalLlmError as error:
        if error.code != "local_llm_unavailable":
            raise

    parsed = urllib.parse.urlparse(resolved_base_url)
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    if host.lower() not in LOCAL_HOSTS:
        raise LocalLlmError(
            code="local_llm_unavailable",
            message=f"Local LLM endpoint is not reachable at {resolved_base_url}",
            suggestion="Start the remote/local LLM server or choose a localhost profile.",
        )
    if _localhost_port_accepts_connection(host, port):
        raise LocalLlmError(
            code="local_llm_port_in_use",
            message=f"Port {host}:{port} is in use, but {resolved_base_url}/models is not reachable.",
            retryable=False,
            suggestion="Choose another base URL port or stop the process using that port.",
        )

    model = normalize_model_name(getattr(profile, "model", ""))
    model_path = find_gguf_model(Path(settings.project_root or "."), model)
    server = find_llama_server(Path(settings.project_root or "."))
    key = f"{host}:{port}:{model_path}"
    _start_server_if_needed(key, server, model_path, model, host, port)
    return _poll_until_server_ready(resolved_base_url, settings, key)


def _start_server_if_needed(key: str, server: Path, model_path: Path, model: str, host: str, port: int) -> None:
    with SERVER_LOCK:
        dead_keys = [item_key for item_key, process in list(SERVER_PROCESSES.items()) if process.poll() is not None]
        for dead_key in dead_keys:
            del SERVER_PROCESSES[dead_key]
        process = SERVER_PROCESSES.get(key)
        if process is None or process.poll() is not None:
            SERVER_PROCESSES[key] = _start_llama_server(server, model_path, model, host, port)


def _poll_until_server_ready(resolved_base_url: str, settings: AssistantExecutionSettings, key: str) -> str:
    deadline = time.time() + min(45, max(10, int(getattr(settings, "timeout_s", 30) or 30)))
    last_error: LocalLlmError | None = None
    while time.time() < deadline:
        try:
            request_json("GET", f"{resolved_base_url}/models", timeout_s=1)
            return f"Started local llama.cpp server at {resolved_base_url}."
        except LocalLlmError as error:
            last_error = error
            with SERVER_LOCK:
                process = SERVER_PROCESSES.get(key)
            if process is not None and process.poll() is not None:
                raise LocalLlmError(
                    code="local_llm_server_exited",
                    message=f"llama-server exited with code {process.returncode}",
                    suggestion="Check the model file, GPU memory, and llama.cpp runtime files.",
                ) from error
            time.sleep(0.5)
    raise LocalLlmError(
        code="local_llm_start_timeout",
        message=f"llama-server did not become ready at {resolved_base_url}",
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


def _localhost_port_accepts_connection(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=0.25):
            return True
    except OSError:
        return False


def _status_timeout(settings: AssistantExecutionSettings) -> int:
    return min(8, max(2, int(getattr(settings, "timeout_s", 6) or 6)))
