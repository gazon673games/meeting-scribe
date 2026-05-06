from __future__ import annotations

from pathlib import Path

from application.model_download_parts.references import is_repo_id, normalize_model_reference


def model_cache_dir(models_dir: str | Path | None = None) -> Path:
    if models_dir:
        return Path(models_dir).expanduser().resolve()
    try:
        from huggingface_hub import constants  # type: ignore

        return Path(constants.HF_HUB_CACHE).expanduser().resolve()
    except Exception:
        return Path("models").resolve()


def hf_repo_dir(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / f"models--{repo_id.replace('/', '--')}"


def resolve_model_root(model_name: str, *, models_dir: str | Path | None = None) -> Path:
    model_ref = normalize_model_reference(model_name)
    direct_path = Path(model_ref).expanduser()
    if direct_path.exists():
        return direct_path.resolve()
    root = model_cache_dir(models_dir)
    if is_repo_id(model_ref):
        for cache_root in candidate_cache_roots(root):
            candidate = hf_repo_dir(cache_root, model_ref)
            if path_exists(candidate):
                return candidate
    return hf_repo_dir(root, model_ref) if is_repo_id(model_ref) else direct_path.resolve()


def candidate_cache_roots(root: Path) -> list[Path]:
    resolved = Path(root).expanduser().resolve()
    candidates = [resolved, resolved / "hub"]
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def path_exists(path: Path) -> bool:
    try:
        return Path(path).exists()
    except OSError:
        return True


def repo_id_from_cache_folder(name: str) -> str:
    raw = str(name or "")
    if not raw.startswith("models--"):
        return ""
    parts = [part.strip() for part in raw[len("models--") :].split("--") if part.strip()]
    if len(parts) < 2:
        return ""
    return f"{parts[0]}/{'--'.join(parts[1:])}"
