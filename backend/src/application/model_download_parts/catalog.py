from __future__ import annotations

import shutil
from pathlib import Path

from application.model_download_parts.constants import BUILTIN_MODEL_REPOS, BUILTIN_MODELS
from application.model_download_parts.inspection import inspect_model_path, looks_like_model_folder
from application.model_download_parts.paths import (
    candidate_cache_roots,
    model_cache_dir,
    path_exists,
    repo_id_from_cache_folder,
    resolve_model_root,
)
from application.model_download_parts.references import normalize_model_reference


def scan_local_models(models_dir: str | Path | None = None) -> list[dict]:
    root = model_cache_dir(models_dir)
    if not path_exists(root):
        return []
    if not root.is_dir():
        raise ValueError(f"Models path is not a folder: {root}")

    records: list[dict] = []
    seen: set[str] = set()
    for index, cache_root in enumerate(candidate_cache_roots(root)):
        if not path_exists(cache_root):
            continue
        for record in scan_one_models_root(cache_root, strict=index == 0):
            key = str(record.get("name") or record.get("path") or "")
            if key and key not in seen:
                seen.add(key)
                records.append(record)
    return records


def delete_local_model(model_name: str, *, models_dir: str | Path | None = None) -> None:
    raw = str(model_name or "").strip()
    if not raw:
        raise ValueError("delete_model requires params.name")
    if raw in BUILTIN_MODELS or raw in set(BUILTIN_MODEL_REPOS.values()):
        raise ValueError("Recommended models are not deleted from the local models view")

    root = model_cache_dir(models_dir)
    model_ref = normalize_model_reference(raw)
    target = resolve_model_root(model_ref, models_dir=root)
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("Can only delete models inside the configured models directory") from exc
    if not target_resolved.exists():
        return
    shutil.rmtree(target_resolved)


def scan_one_models_root(root: Path, *, strict: bool) -> list[dict]:
    try:
        children = sorted(root.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        if strict:
            raise
        return [inaccessible_model_record(root, exc)]

    records: list[dict] = []
    for child in children:
        try:
            is_dir = child.is_dir()
        except OSError as exc:
            records.append(inaccessible_model_record(child, exc))
            continue
        if not is_dir:
            continue
        try:
            if child.name.startswith("models--"):
                repo_id = repo_id_from_cache_folder(child.name)
                if not repo_id:
                    continue
                record = local_model_record(repo_id, child, source="huggingface-cache")
            elif looks_like_model_folder(child):
                record = local_model_record(str(child), child, source="local-folder", label=child.name)
            else:
                continue
        except OSError as exc:
            record = inaccessible_model_record(child, exc)
        records.append(record)
    return records


def local_model_record(name: str, path: Path, *, source: str, label: str = "") -> dict:
    info = inspect_model_path(path)
    return {
        "name": name,
        "label": label or name,
        "cached": bool(info["compatible"]),
        "compatible": bool(info["compatible"]),
        "status": str(info["status"]),
        "source": source,
        "path": str(info.get("rootPath") or path),
        "resolvedPath": str(info.get("path") or path),
        "missing": list(info.get("missing") or []),
        "warnings": list(info.get("warnings") or []),
        "builtin": False,
        "recommended": False,
        "downloadable": False,
        "deletable": True,
    }


def inaccessible_model_record(path: Path, error: Exception) -> dict:
    return {
        "name": str(path),
        "label": path.name,
        "cached": False,
        "compatible": False,
        "status": "inaccessible",
        "source": "local-folder",
        "path": str(path),
        "resolvedPath": str(path),
        "missing": [],
        "warnings": [f"{type(error).__name__}: {error}"],
        "builtin": False,
        "recommended": False,
        "downloadable": False,
        "deletable": False,
    }
