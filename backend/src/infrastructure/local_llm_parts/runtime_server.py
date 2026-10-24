from __future__ import annotations

import logging
import os
import socket
import subprocess
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from application.codex_config import CodexProfile
from application.codex_prompting import normalize_model_name
from assistant.application.provider import AssistantExecutionSettings
from infrastructure.local_llm_parts.errors import LocalLlmError
from infrastructure.local_llm_parts.http_client import request_json
from infrastructure.local_llm_parts.profile_utils import base_url
from infrastructure.local_llm_parts.runtime_discovery import (
    find_gguf_model,
    find_llama_server,
)

LOCAL_HOSTS = frozenset(('127.0.0.1', 'localhost', '::1'))
SERVER_LOCK = threading.RLock()


class ServerKey(NamedTuple):
    host: str
    port: int
    model_path: Path


SERVER_PROCESSES: dict[ServerKey, subprocess.Popen] = {}


@dataclass
class _ServerStart:
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None


_STARTS: dict[tuple[str, int], _ServerStart] = {}
_logger = logging.getLogger(__name__)


def start_local_llm_async(
    profile: CodexProfile,
    project_root: Path,
    emit_event: Callable[[dict], None],
    *,
    default_base_url: str,
) -> dict:
    """Start llama-server for an openai_local profile in a background thread."""
    profile_id = str(profile.id or profile.label or 'local')
    resolved_base_url = base_url(profile, default_base_url)

    settings = AssistantExecutionSettings(
        command_tokens=[],
        path_hints=[],
        proxy='',
        timeout_s=60,
        project_root=project_root,
    )
    prefix = _server_prefix(resolved_base_url)
    with SERVER_LOCK:
        if prefix in _STARTS:
            return {'started': False, 'profileId': profile_id}
        job = _ServerStart()
        _STARTS[prefix] = job

    def _run() -> None:
        def _emit(state: str, message: str = '') -> None:
            if callable(emit_event):
                emit_event(
                    {
                        'type': 'local_llm_status',
                        'profileId': profile_id,
                        'state': state,
                        'message': message,
                    }
                )

        try:
            _emit('starting')
            message = ensure_openai_local_runtime(
                profile, settings, resolved_base_url, cancelled=job.cancelled
            )
            if job.cancelled.is_set():
                _emit('stopped', 'Startup cancelled')
            else:
                _emit('running', message)
        except LocalLlmError as error:
            _emit(
                'stopped' if job.cancelled.is_set() else 'error', error.message
            )
        except Exception as error:  # pragma: no cover - safety net
            _emit('error', str(error))
        finally:
            with SERVER_LOCK:
                if _STARTS.get(prefix) is job:
                    del _STARTS[prefix]
            job.done.set()

    try:
        job.worker = threading.Thread(
            target=_run, daemon=True, name=f'llm-start-{profile_id}'
        )
        job.worker.start()
    except BaseException:
        with SERVER_LOCK:
            if _STARTS.get(prefix) is job:
                del _STARTS[prefix]
        job.done.set()
        raise
    return {'started': True, 'profileId': profile_id}


def stop_local_llm(profile: CodexProfile, *, default_base_url: str) -> dict:
    """Terminate llama-server started for a given profile base_url host:port."""
    profile_id = str(profile.id or profile.label or 'local')
    resolved_base_url = base_url(profile, default_base_url)
    prefix = _server_prefix(resolved_base_url)

    killed = 0
    with SERVER_LOCK:
        job = _STARTS.get(prefix)
        if job is not None:
            job.cancelled.set()
        processes = [
            (key, process)
            for key, process in SERVER_PROCESSES.items()
            if key[:2] == prefix
        ]
    failures: list[Exception] = []
    for key, process in processes:
        try:
            killed += int(_stop_process(key, process))
        except (OSError, subprocess.SubprocessError) as error:
            failures.append(error)
            continue
    if failures:
        raise LocalLlmError(
            code='local_llm_stop_failed',
            message=f'Cannot stop local LLM process: {failures[0]}',
        ) from failures[0]

    return {'stopped': True, 'profileId': profile_id, 'killed': killed}


def ensure_openai_local_runtime(
    profile: CodexProfile,
    settings: AssistantExecutionSettings,
    resolved_base_url: str,
    *,
    cancelled: threading.Event | None = None,
) -> str:
    _check_cancelled(cancelled)
    try:
        request_json(
            'GET',
            f'{resolved_base_url}/models',
            timeout_s=_status_timeout(settings),
        )
        return f'Local endpoint is reachable at {resolved_base_url}.'
    except LocalLlmError as error:
        if error.code != 'local_llm_unavailable':
            raise

    parsed = urllib.parse.urlparse(resolved_base_url)
    host = parsed.hostname or '127.0.0.1'
    port = int(parsed.port or (443 if parsed.scheme == 'https' else 80))
    if host.lower() not in LOCAL_HOSTS:
        raise LocalLlmError(
            code='local_llm_unavailable',
            message=f'Local LLM endpoint is not reachable at {resolved_base_url}',
            suggestion='Start the remote/local LLM server or choose a localhost profile.',
        )
    if _localhost_port_accepts_connection(host, port):
        raise LocalLlmError(
            code='local_llm_port_in_use',
            message=f'Port {host}:{port} is in use, but {resolved_base_url}/models is not reachable.',
            retryable=False,
            suggestion='Choose another base URL port or stop the process using that port.',
        )

    model = normalize_model_name(profile.model)
    model_path = find_gguf_model(Path(settings.project_root or '.'), model)
    server = find_llama_server(Path(settings.project_root or '.'))
    key = ServerKey(host, port, model_path)
    process = _start_server_if_needed(
        key, server, model_path, model, host, port, cancelled=cancelled
    )
    try:
        return _poll_until_server_ready(
            resolved_base_url, settings, key, cancelled=cancelled
        )
    except BaseException:
        if process is not None:
            try:
                _stop_process(key, process)
            except Exception:
                _logger.exception('Cannot stop failed local LLM startup')
        raise


def _start_server_if_needed(
    key: ServerKey,
    server: Path,
    model_path: Path,
    model: str,
    host: str,
    port: int,
    *,
    cancelled: threading.Event | None = None,
) -> subprocess.Popen | None:
    with SERVER_LOCK:
        _check_cancelled(cancelled)
        dead_keys = [
            item_key
            for item_key, process in list(SERVER_PROCESSES.items())
            if process.poll() is not None
        ]
        for dead_key in dead_keys:
            del SERVER_PROCESSES[dead_key]
        process = SERVER_PROCESSES.get(key)
        if process is None or process.poll() is not None:
            process = _start_llama_server(server, model_path, model, host, port)
            SERVER_PROCESSES[key] = process
            return process
    return None


def _stop_process(key: ServerKey, process: subprocess.Popen) -> bool:
    running = process.poll() is None
    if running:
        process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
    with SERVER_LOCK:
        if SERVER_PROCESSES.get(key) is process:
            del SERVER_PROCESSES[key]
    return running


def shutdown() -> None:
    with SERVER_LOCK:
        jobs = list(_STARTS.values())
        for job in jobs:
            job.cancelled.set()
    for job in jobs:
        if job.worker is threading.current_thread():
            continue
        job.done.wait()
        if (
            job.worker is not None
            and job.worker is not threading.current_thread()
        ):
            job.worker.join()
    with SERVER_LOCK:
        processes = list(SERVER_PROCESSES.items())
    failures: list[Exception] = []
    for key, process in processes:
        try:
            _stop_process(key, process)
        except Exception as error:
            failures.append(error)
            _logger.exception('Cannot stop local LLM server %s', key)
    if failures:
        raise failures[0]


def _poll_until_server_ready(
    resolved_base_url: str,
    settings: AssistantExecutionSettings,
    key: ServerKey,
    *,
    cancelled: threading.Event | None = None,
) -> str:
    deadline = time.monotonic() + min(
        45, max(10, int(settings.timeout_s or 30))
    )
    last_error: LocalLlmError | None = None
    while time.monotonic() < deadline:
        _check_cancelled(cancelled)
        try:
            request_json('GET', f'{resolved_base_url}/models', timeout_s=1)
            return f'Started local llama.cpp server at {resolved_base_url}.'
        except LocalLlmError as error:
            last_error = error
            with SERVER_LOCK:
                process = SERVER_PROCESSES.get(key)
            if process is not None and process.poll() is not None:
                raise LocalLlmError(
                    code='local_llm_server_exited',
                    message=f'llama-server exited with code {process.returncode}',
                    suggestion='Check the model file, GPU memory, and llama.cpp runtime files.',
                ) from error
            if cancelled is None:
                time.sleep(0.5)
            else:
                cancelled.wait(0.5)
    raise LocalLlmError(
        code='local_llm_start_timeout',
        message=f'llama-server did not become ready at {resolved_base_url}',
        suggestion=(
            last_error.suggestion
            if last_error
            else 'Check the local LLM server logs.'
        ),
    )


def _start_llama_server(
    server: Path, model_path: Path, alias: str, host: str, port: int
) -> subprocess.Popen:
    command = [
        str(server),
        '-m',
        str(model_path),
        '--host',
        host,
        '--port',
        str(port),
        '-a',
        alias,
        '-c',
        os.environ.get('LLAMA_SERVER_CTX', '4096'),
    ]
    ngl = str(os.environ.get('LLAMA_SERVER_NGL', '28')).strip()
    if ngl:
        command.extend(['-ngl', ngl])

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
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
    return min(8, max(2, int(settings.timeout_s or 6)))


def _check_cancelled(cancelled: threading.Event | None) -> None:
    if cancelled is not None and cancelled.is_set():
        raise LocalLlmError(
            code='local_llm_start_cancelled',
            message='Local LLM startup cancelled',
        )


def _server_prefix(url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or '127.0.0.1').lower()
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    return host, port


def model_path_in_use(path: Path) -> bool:
    wanted = path.expanduser().resolve()
    with SERVER_LOCK:
        keys = list(SERVER_PROCESSES)
    return any(key.model_path.resolve() == wanted for key in keys)
