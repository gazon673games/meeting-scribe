from __future__ import annotations

import threading
from typing import Callable, Optional


BUILTIN_MODELS = {"tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo"}


def is_model_cached(model_name: str) -> bool:
    """Return True if the model weights are fully present in the local HuggingFace cache."""
    if model_name in BUILTIN_MODELS:
        return True
    try:
        from huggingface_hub import constants  # type: ignore
        import pathlib

        repo_id = model_name.replace("\\", "/")
        cache_dir = pathlib.Path(constants.HF_HUB_CACHE)
        folder = repo_id.replace("/", "--")
        repo_dir = cache_dir / f"models--{folder}"
        if not repo_dir.exists():
            return False

        weight_names = {"model.bin", "model.safetensors"}
        min_size = 1_000_000  # 1 MB — configs are tiny, weights are large
        for p in repo_dir.rglob("*"):
            if p.name in weight_names and p.stat().st_size > min_size:
                return True
        return False
    except Exception:
        return False


def download_model_async(
    model_name: str,
    on_progress: Callable[[str], None],
    on_done: Callable[[Optional[str]], None],
) -> None:
    """Download model in a background thread.

    Calls on_progress(message) with status updates.
    Calls on_done(None) on success or on_done(error_str) on failure.
    """
    def _run() -> None:
        try:
            on_progress(f"Downloading {model_name}...")
            from huggingface_hub import snapshot_download  # type: ignore
            snapshot_download(
                repo_id=model_name,
                local_files_only=False,
            )
            on_done(None)
        except Exception as e:
            on_done(str(e))

    t = threading.Thread(target=_run, name="model-download", daemon=True)
    t.start()
