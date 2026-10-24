from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from application.codex_config import CodexProfile
from assistant.application.provider import (
    ASSISTANT_PROVIDER_LOCAL,
    AssistantExecutionSettings,
    AssistantProviderError,
    AssistantProviderInfo,
    AssistantProviderPingResult,
    AssistantProviderRequest,
    AssistantProviderResult,
    result_from_error,
)

_LOCK = threading.RLock()
_CANCELLED: set[str] = set()
_MODEL_GUARDS: dict[str, AbstractContextManager] = {}


@dataclass
class _LoadJob:
    model_name: str
    project_root: Path | None
    cancelled: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    worker: threading.Thread | None = None


_LOADS: dict[str, _LoadJob] = {}


@dataclass(frozen=True)
class _LoadedModel:
    engine: Any
    path: Path | None


_LOADED: dict[str, _LoadedModel] = {}


_logger = logging.getLogger(__name__)


class LlamaCppRunner:
    provider_id = ASSISTANT_PROVIDER_LOCAL
    provider_label = 'Local GGUF'

    def status(
        self, settings: AssistantExecutionSettings
    ) -> AssistantProviderInfo:
        profile = settings.profile
        key = str(profile.id or '') if profile is not None else ''
        with _LOCK:
            loaded = bool(key and key in _LOADED)
        return AssistantProviderInfo(
            id=self.provider_id,
            label=self.provider_label,
            available=loaded,
            message='Model loaded and ready.'
            if loaded
            else 'Model not loaded.',
            suggestion=''
            if loaded
            else 'Click Start Model to load the GGUF model.',
        )

    def ping(
        self, settings: AssistantExecutionSettings
    ) -> AssistantProviderPingResult:
        profile = settings.profile
        key = str(profile.id or '') if profile is not None else ''
        with _LOCK:
            loaded = bool(key and key in _LOADED)
        if loaded:
            model_name = str(profile.model or '')
            return AssistantProviderPingResult(
                id=self.provider_id,
                label=self.provider_label,
                ok=True,
                message=f'Model loaded: {model_name}',
            )
        return AssistantProviderPingResult(
            id=self.provider_id,
            label=self.provider_label,
            ok=False,
            message='Model not loaded.',
            error_code='model_not_loaded',
            retryable=False,
            suggestion='Click Start Model to load the GGUF model.',
        )

    def run(self, request: AssistantProviderRequest) -> AssistantProviderResult:
        # Unload and inference share a model-specific lease, not the registry lock.
        key = str(request.profile.id or '')
        with _model_guard(key):
            return self._run_loaded(request)

    def _run_loaded(
        self, request: AssistantProviderRequest
    ) -> AssistantProviderResult:
        started = time.time()
        profile = request.profile
        key = str(profile.id or '')
        with _LOCK:
            resource = _LOADED.get(key)

        if resource is None:
            return result_from_error(
                profile=profile,
                cmd=request.original_cmd,
                provider=self.provider_id,
                model=str(profile.model),
                error=AssistantProviderError(
                    code='model_not_loaded',
                    message='Model is not loaded. Click Start Model first.',
                    retryable=False,
                    suggestion='Click Start Model to load the GGUF model before running.',
                ),
                started_at=started,
            )

        try:
            llm = resource.engine
            messages = [{'role': 'user', 'content': request.prompt}]
            kwargs: dict[str, Any] = {'messages': messages, 'stream': False}
            temp = _optional_float(profile.temperature)
            max_tok = max(0, int(profile.max_tokens or 0))
            if temp is not None:
                kwargs['temperature'] = temp
            if max_tok > 0:
                kwargs['max_tokens'] = max_tok
            response = llm.create_chat_completion(**kwargs)
            text = (
                response['choices'][0]['message']['content'] or ''
            ).strip() or '(empty response)'
            return AssistantProviderResult(
                ok=True,
                profile=str(profile.label),
                cmd=request.original_cmd,
                text=text,
                dt_s=time.time() - started,
                provider=self.provider_id,
                model=str(profile.model),
            )
        except Exception as error:
            return result_from_error(
                profile=profile,
                cmd=request.original_cmd,
                provider=self.provider_id,
                model=str(profile.model),
                error=AssistantProviderError(
                    code='inference_error',
                    message=f'{type(error).__name__}: {error}',
                    retryable=True,
                ),
                started_at=started,
            )


def load_model_async(
    profile: CodexProfile,
    project_root: Path | None,
    emit_event: Callable[[dict], None],
) -> dict:
    profile_id = str(profile.id or 'local')

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

    def _run() -> None:
        try:
            _emit('starting', 'Loading model into memory...')
            if job.cancelled.is_set():
                _emit('stopped', 'Load cancelled')
                return
            llm = _load_model(
                profile, Path(project_root) if project_root else None
            )
            with _model_guard(profile_id):
                with _LOCK:
                    cancelled = (
                        job.cancelled.is_set() or profile_id in _CANCELLED
                    )
                    # Retain ownership even if closing a cancelled model fails.
                    _LOADED[profile_id] = llm
                if cancelled:
                    llm.engine.close()
                    with _LOCK:
                        del _LOADED[profile_id]
                    _emit('stopped', 'Load cancelled')
                    return
                model_name = Path(profile.model or '').name or profile.model
                _emit('running', f'Ready: {model_name}')
        except Exception as error:
            _emit('error', str(error))
        finally:
            with _LOCK:
                if _LOADS.get(profile_id) is job:
                    del _LOADS[profile_id]
            job.done.set()

    with _LOCK:
        loaded = profile_id in _LOADED
        if profile_id in _LOADS:
            return {'started': False, 'profileId': profile_id}
        if not loaded:
            job = _LoadJob(profile.model, project_root)
            _LOADS[profile_id] = job
            _CANCELLED.discard(profile_id)
    if loaded:
        _emit('running', 'Model already loaded')
        return {'started': True, 'profileId': profile_id}
    try:
        job.worker = threading.Thread(
            target=_run, daemon=True, name=f'llm-load-{profile_id}'
        )
        job.worker.start()
    except BaseException:
        with _LOCK:
            if _LOADS.get(profile_id) is job:
                del _LOADS[profile_id]
        job.done.set()
        raise
    return {'started': True, 'profileId': profile_id}


def unload_model(profile: CodexProfile) -> dict:
    profile_id = str(profile.id or 'local')
    _unload_model_id(profile_id)
    return {'stopped': True, 'profileId': profile_id}


def _unload_model_id(profile_id: str) -> None:
    with _LOCK:
        _CANCELLED.add(profile_id)
        job = _LOADS.get(profile_id)
        if job is not None:
            job.cancelled.set()
    with _model_guard(profile_id):
        with _LOCK:
            llm = _LOADED.get(profile_id)
        if llm is not None:
            llm.engine.close()
            with _LOCK:
                if _LOADED.get(profile_id) is llm:
                    del _LOADED[profile_id]


def shutdown() -> None:
    with _LOCK:
        jobs = list(_LOADS.values())
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
    with _LOCK:
        profile_ids = list(_LOADED)
    failures: list[Exception] = []
    for profile_id in profile_ids:
        try:
            _unload_model_id(profile_id)
        except Exception as error:
            failures.append(error)
            _logger.exception('Cannot unload local model %s', profile_id)
    if failures:
        raise failures[0]


def _model_guard(profile_id: str) -> AbstractContextManager:
    with _LOCK:
        guard = _MODEL_GUARDS.get(profile_id)
        if guard is None:
            guard = threading.RLock()
            _MODEL_GUARDS[profile_id] = guard
        return guard


def model_path_in_use(path: Path) -> bool:
    wanted = path.expanduser().resolve()
    with _LOCK:
        resources = list(_LOADED.values())
        pending = [
            (job.model_name, job.project_root) for job in _LOADS.values()
        ]
    if any(resource.path == wanted for resource in resources):
        return True
    for name, root in pending:
        try:
            candidate = _resolve_model_path(name, root)
        except FileNotFoundError:
            continue
        if candidate == wanted:
            return True
    return False


def _load_model(
    profile: CodexProfile, project_root: Path | None
) -> _LoadedModel:
    try:
        from llama_cpp import Llama  # type: ignore[import]
    except ImportError as error:
        raise RuntimeError(
            'llama-cpp-python is not installed. '
            'Run: pip install llama-cpp-python  '
            '(or pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 for CUDA)'
        ) from error

    model = str(profile.model or '').strip()
    if not model:
        raise ValueError(
            "Profile 'model' field is required — set a path to a .gguf file."
        )

    model_path = _resolve_model_path(model, project_root)
    n_gpu_layers = max(0, int(profile.gpu_layers or 0))
    n_ctx = max(64, int(profile.context_size or 4096))

    return _LoadedModel(
        engine=Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            verbose=False,
        ),
        path=model_path,
    )


def _search_models_dir(models_dir: Path, model: str) -> Path | None:
    if not models_dir.exists():
        return None
    stem = Path(model).stem
    for m in sorted(models_dir.rglob('*.gguf'), key=lambda p: str(p).lower()):
        if m.stem == stem or m.name == model:
            return m.resolve()
    return None


def _resolve_model_path(model: str, project_root: Path | None) -> Path:
    direct = Path(model).expanduser()
    if direct.is_absolute() and direct.exists():
        return direct.resolve()

    candidates: list[Path] = [direct]
    if project_root:
        candidates.append(project_root / model)
        candidates.append(project_root / 'models' / 'llm' / model)
        if not model.lower().endswith('.gguf'):
            candidates.append(
                project_root / 'models' / 'llm' / (model + '.gguf')
            )

    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()

    if project_root:
        found = _search_models_dir(project_root / 'models' / 'llm', model)
        if found is not None:
            return found

    raise FileNotFoundError(
        f"GGUF model '{model}' not found. "
        "Set 'model' in profile to a full path or place the .gguf file in models/llm/."
    )


def _optional_float(raw: Any) -> float | None:
    if raw is None or str(raw).strip() == '':
        return None
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError, OverflowError):
        return None
