from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

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
_LOADED: dict[str, Any] = {}
_CANCELLED: set[str] = set()


class LlamaCppRunner:
    provider_id = ASSISTANT_PROVIDER_LOCAL
    provider_label = "Local GGUF"

    def status(self, settings: AssistantExecutionSettings) -> AssistantProviderInfo:
        profile = getattr(settings, "profile", None)
        key = str(getattr(profile, "id", "") or "") if profile else ""
        with _LOCK:
            loaded = bool(key and key in _LOADED)
        return AssistantProviderInfo(
            id=self.provider_id,
            label=self.provider_label,
            available=loaded,
            message="Model loaded and ready." if loaded else "Model not loaded.",
            suggestion="" if loaded else "Click Start Model to load the GGUF model.",
        )

    def ping(self, settings: AssistantExecutionSettings) -> AssistantProviderPingResult:
        profile = getattr(settings, "profile", None)
        key = str(getattr(profile, "id", "") or "") if profile else ""
        with _LOCK:
            loaded = bool(key and key in _LOADED)
        if loaded:
            model_name = str(getattr(profile, "model", "") or "")
            return AssistantProviderPingResult(
                id=self.provider_id,
                label=self.provider_label,
                ok=True,
                message=f"Model loaded: {model_name}",
            )
        return AssistantProviderPingResult(
            id=self.provider_id,
            label=self.provider_label,
            ok=False,
            message="Model not loaded.",
            error_code="model_not_loaded",
            retryable=False,
            suggestion="Click Start Model to load the GGUF model.",
        )

    def run(self, request: AssistantProviderRequest) -> AssistantProviderResult:
        started = time.time()
        profile = request.profile
        key = str(getattr(profile, "id", "") or "")
        with _LOCK:
            llm = _LOADED.get(key)

        if llm is None:
            return result_from_error(
                profile=profile,
                cmd=request.original_cmd,
                provider=self.provider_id,
                model=str(getattr(profile, "model", "")),
                error=AssistantProviderError(
                    code="model_not_loaded",
                    message="Model is not loaded. Click Start Model first.",
                    retryable=False,
                    suggestion="Click Start Model to load the GGUF model before running.",
                ),
                started_at=started,
            )

        try:
            messages = [{"role": "user", "content": request.prompt}]
            kwargs: dict[str, Any] = {"messages": messages, "stream": False}
            temp = _optional_float(getattr(profile, "temperature", None))
            max_tok = max(0, int(getattr(profile, "max_tokens", 0) or 0))
            if temp is not None:
                kwargs["temperature"] = temp
            if max_tok > 0:
                kwargs["max_tokens"] = max_tok
            response = llm.create_chat_completion(**kwargs)
            text = (response["choices"][0]["message"]["content"] or "").strip() or "(empty response)"
            return AssistantProviderResult(
                ok=True,
                profile=str(getattr(profile, "label", "")),
                cmd=request.original_cmd,
                text=text,
                dt_s=time.time() - started,
                provider=self.provider_id,
                model=str(getattr(profile, "model", "")),
            )
        except Exception as exc:
            return result_from_error(
                profile=profile,
                cmd=request.original_cmd,
                provider=self.provider_id,
                model=str(getattr(profile, "model", "")),
                error=AssistantProviderError(
                    code="inference_error",
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                ),
                started_at=started,
            )


def load_model_async(profile: Any, project_root: Any, emit_event: Callable[[dict], None]) -> dict:
    profile_id = str(getattr(profile, "id", "") or "local")

    def _emit(state: str, message: str = "") -> None:
        if callable(emit_event):
            emit_event({"type": "local_llm_status", "profileId": profile_id, "state": state, "message": message})

    def _run() -> None:
        with _LOCK:
            if profile_id in _LOADED:
                model_name = Path(getattr(profile, "model", "") or "").name or getattr(profile, "model", "")
                _emit("running", f"Model already loaded: {model_name}")
                return
            _CANCELLED.discard(profile_id)

        _emit("starting", "Loading model into memory...")
        try:
            llm = _load_model(profile, Path(project_root) if project_root else None)
            with _LOCK:
                if profile_id in _CANCELLED:
                    # unload_model was called while we were loading — discard immediately
                    _CANCELLED.discard(profile_id)
                    try:
                        llm.close()
                    except Exception:
                        pass
                    _emit("stopped", "Load cancelled")
                    return
                _LOADED[profile_id] = llm
            model_name = Path(getattr(profile, "model", "") or "").name or getattr(profile, "model", "")
            _emit("running", f"Ready: {model_name}")
        except Exception as exc:
            _emit("error", str(exc))

    threading.Thread(target=_run, daemon=True, name=f"llm-load-{profile_id}").start()
    return {"started": True, "profileId": profile_id}


def unload_model(profile: Any) -> dict:
    profile_id = str(getattr(profile, "id", "") or "local")
    with _LOCK:
        llm = _LOADED.pop(profile_id, None)
        # Signal any in-progress load to discard the model when done
        _CANCELLED.add(profile_id)
    if llm is not None:
        try:
            llm.close()
        except Exception:
            pass
    return {"stopped": True, "profileId": profile_id}


def _load_model(profile: Any, project_root: Path | None) -> Any:
    try:
        from llama_cpp import Llama  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "llama-cpp-python is not installed. "
            "Run: pip install llama-cpp-python  "
            "(or pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 for CUDA)"
        ) from exc

    model = str(getattr(profile, "model", "") or "").strip()
    if not model:
        raise ValueError("Profile 'model' field is required — set a path to a .gguf file.")

    model_path = _resolve_model_path(model, project_root)
    n_gpu_layers = max(0, int(getattr(profile, "gpu_layers", 0) or 0))
    n_ctx = max(64, int(getattr(profile, "context_size", 4096) or 4096))

    return Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False,
    )


def _search_models_dir(models_dir: Path, model: str) -> Path | None:
    if not models_dir.exists():
        return None
    stem = Path(model).stem
    for m in sorted(models_dir.rglob("*.gguf"), key=lambda p: str(p).lower()):
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
        candidates.append(project_root / "models" / "llm" / model)
        if not model.lower().endswith(".gguf"):
            candidates.append(project_root / "models" / "llm" / (model + ".gguf"))

    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()

    if project_root:
        found = _search_models_dir(project_root / "models" / "llm", model)
        if found is not None:
            return found

    raise FileNotFoundError(
        f"GGUF model '{model}' not found. "
        "Set 'model' in profile to a full path or place the .gguf file in models/llm/."
    )


def _optional_float(raw: Any) -> float | None:
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(str(raw).strip())
    except Exception:
        return None
