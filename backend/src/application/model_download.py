from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional
from urllib.parse import urlparse


BUILTIN_MODELS = {"tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo"}

_BUILTIN_MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large": "Systran/faster-whisper-large",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",
}

_HF_REPO_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_DOWNLOAD_ENV_LOCK = threading.RLock()


def normalize_model_reference(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc.lower().endswith("huggingface.co"):
        parts = [part for part in parsed.path.strip("/").split("/") if part and part not in {"tree", "resolve"}]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return _BUILTIN_MODEL_REPOS.get(text, text.replace("\\", "/"))


def is_valid_huggingface_repo_id(value: str) -> bool:
    text = str(value or "").strip()
    if not text or "://" in text or "\\" in text or ":" in text:
        return False
    parts = text.split("/")
    if len(parts) != 2 or any(not part for part in parts):
        return False
    for part in parts:
        if part in {".", ".."} or part.startswith(("-", ".")) or part.endswith(("-", ".")):
            return False
        if ".." in part or not _HF_REPO_SEGMENT_RE.match(part):
            return False
    return True


def model_cache_dir(models_dir: str | Path | None = None) -> Path:
    if models_dir:
        return Path(models_dir).expanduser().resolve()
    try:
        from huggingface_hub import constants  # type: ignore

        return Path(constants.HF_HUB_CACHE).expanduser().resolve()
    except Exception:
        return Path("models").resolve()


def is_model_cached(model_name: str, *, models_dir: str | Path | None = None) -> bool:
    """Return True if the model weights are fully present in the local HuggingFace cache."""
    model_ref = normalize_model_reference(model_name)
    if not model_ref:
        return False
    try:
        return inspect_model_path(_resolve_model_root(model_ref, models_dir=models_dir))["compatible"]
    except Exception:
        return False


def is_builtin_model(model_name: str) -> bool:
    return model_name in BUILTIN_MODELS


def scan_local_models(models_dir: str | Path | None = None) -> list[dict]:
    root = model_cache_dir(models_dir)
    if not _path_exists(root):
        return []
    if not root.is_dir():
        raise ValueError(f"Models path is not a folder: {root}")

    records: list[dict] = []
    seen: set[str] = set()
    for index, cache_root in enumerate(_candidate_cache_roots(root)):
        if not _path_exists(cache_root):
            continue
        for record in _scan_one_models_root(cache_root, strict=index == 0):
            key = str(record.get("name") or record.get("path") or "")
            if key and key not in seen:
                seen.add(key)
                records.append(record)
    return records


def delete_local_model(model_name: str, *, models_dir: str | Path | None = None) -> None:
    raw = str(model_name or "").strip()
    if not raw:
        raise ValueError("delete_model requires params.name")
    if raw in BUILTIN_MODELS or raw in set(_BUILTIN_MODEL_REPOS.values()):
        raise ValueError("Recommended models are not deleted from the local models view")

    root = model_cache_dir(models_dir)
    model_ref = normalize_model_reference(raw)
    target = _resolve_model_root(model_ref, models_dir=root)
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    try:
        target_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("Can only delete models inside the configured models directory") from exc
    if not target_resolved.exists():
        return
    shutil.rmtree(target_resolved)


def inspect_model_path(path: str | Path) -> dict:
    raw_path = Path(path).expanduser()
    target = _best_model_dir(raw_path)
    missing: list[str] = []
    warnings: list[str] = []
    if not _path_exists(raw_path):
        return {"compatible": False, "status": "missing", "path": str(raw_path), "missing": ["folder"]}
    if not (target / "config.json").exists():
        missing.append("config.json")
    if not _has_ctranslate2_weight_file(target):
        if _has_transformers_weight_file(target):
            missing.append("model.bin")
            warnings.append("Transformers/safetensors weights are not supported by faster-whisper; use a CTranslate2 model.bin export.")
        else:
            missing.append("model.bin")
    if not _has_tokenizer_file(target):
        missing.append("tokenizer.json or vocabulary.json")
    compatible = not missing
    if compatible:
        status = "compatible"
        model_format = "ctranslate2"
    elif _has_transformers_weight_file(target):
        status = "unsupported_transformers_format"
        model_format = "transformers"
    else:
        status = "missing_files"
        model_format = "unknown"
    return {
        "compatible": compatible,
        "status": status,
        "format": model_format,
        "path": str(target),
        "rootPath": str(raw_path),
        "missing": missing,
        "warnings": warnings,
    }


def model_metadata(model_name: str, *, models_dir: str | Path | None = None) -> dict:
    raw = str(model_name or "").strip()
    model_ref = normalize_model_reference(raw)
    root = model_cache_dir(models_dir)
    model_root = _resolve_model_root(model_ref, models_dir=root)
    info = inspect_model_path(model_root)
    resolved = Path(str(info.get("path") or model_root))
    model_root_exists = model_root.exists()
    direct_path = Path(model_ref).expanduser()
    return {
        "name": raw,
        "normalizedName": model_ref,
        "repoId": model_ref if _is_repo_id(model_ref) else "",
        "source": "local-folder" if direct_path.exists() else "huggingface-cache",
        "builtin": raw in BUILTIN_MODELS or model_ref in set(_BUILTIN_MODEL_REPOS.values()),
        "cacheDir": str(root),
        "cachePath": str(model_root),
        "resolvedPath": str(resolved),
        "cached": bool(info.get("compatible", False)),
        "compatible": bool(info.get("compatible", False)),
        "status": str(info.get("status") or "unknown"),
        "format": str(info.get("format") or "unknown"),
        "missing": list(info.get("missing") or []),
        "warnings": list(info.get("warnings") or []),
        "totalBytes": _directory_size(model_root) if model_root_exists else 0,
        "presentFiles": _present_model_files(resolved) if resolved.exists() else [],
        "weightFiles": _weight_file_records(resolved) if resolved.exists() else [],
        "config": _read_json_summary(
            resolved / "config.json",
            [
                "model_type",
                "architectures",
                "is_encoder_decoder",
                "num_mel_bins",
                "d_model",
                "encoder_layers",
                "decoder_layers",
                "encoder_attention_heads",
                "decoder_attention_heads",
                "vocab_size",
                "max_source_positions",
                "max_target_positions",
            ],
        ),
        "preprocessor": _read_json_summary(
            resolved / "preprocessor_config.json",
            [
                "feature_extractor_type",
                "sampling_rate",
                "chunk_length",
                "n_fft",
                "hop_length",
                "feature_size",
                "num_mel_bins",
            ],
        ),
        "tokenizer": _read_json_summary(
            resolved / "tokenizer_config.json",
            [
                "tokenizer_class",
                "model_max_length",
                "language",
                "task",
                "normalize",
                "bos_token",
                "eos_token",
                "unk_token",
                "pad_token",
            ],
        ),
        "readme": _read_model_card_metadata(model_root),
    }


def download_model_async(
    model_name: str,
    on_progress: Callable[[dict], None],
    on_done: Callable[[Optional[str]], None],
    *,
    models_dir: str | Path | None = None,
    proxy: str = "",
) -> None:
    """Download model in a background thread.

    Calls on_progress(payload) with status updates.
    Calls on_done(None) on success or on_done(error_str) on failure.
    """
    def _run() -> None:
        stop_event: threading.Event | None = None
        monitor: threading.Thread | None = None
        try:
            model_ref = normalize_model_reference(model_name)
            if not model_ref:
                raise ValueError("Model name is empty")
            if Path(model_ref).expanduser().exists():
                raise ValueError("Local model folders do not need downloading")
            if not is_valid_huggingface_repo_id(model_ref):
                raise ValueError(f"Invalid Hugging Face repo id: {model_ref}")
            cache_dir = model_cache_dir(models_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            repo_dir = _hf_repo_dir(cache_dir, model_ref)
            on_progress(
                {
                    "message": f"Downloading {model_ref}...",
                    "downloadedBytes": _directory_size(repo_dir),
                    "speedBps": 0,
                }
            )
            stop_event = threading.Event()
            monitor = threading.Thread(
                target=_monitor_download_progress,
                args=(repo_dir, stop_event, on_progress, model_ref),
                name="model-download-progress",
                daemon=True,
            )
            monitor.start()
            from huggingface_hub import snapshot_download  # type: ignore

            with _temporary_proxy_env(proxy):
                snapshot_download(
                    repo_id=model_ref,
                    cache_dir=str(cache_dir),
                    local_files_only=False,
                )
            on_progress(
                {
                    "message": "Validating downloaded model...",
                    "downloadedBytes": _directory_size(repo_dir),
                    "speedBps": 0,
                }
            )
            info = inspect_model_path(repo_dir)
            if not info.get("compatible", False):
                missing = ", ".join(str(item) for item in info.get("missing", [])) or "compatible CTranslate2 files"
                raise ValueError(f"Downloaded model is not faster-whisper compatible: missing {missing}")
            on_progress(
                {
                    "message": "Downloaded",
                    "downloadedBytes": _directory_size(repo_dir),
                    "speedBps": 0,
                }
            )
            on_done(None)
        except Exception as e:
            on_done(_friendly_download_error(e))
        finally:
            if stop_event is not None:
                stop_event.set()
            if monitor is not None:
                monitor.join(timeout=0.5)

    t = threading.Thread(target=_run, name="model-download", daemon=True)
    t.start()


def _monitor_download_progress(
    repo_dir: Path,
    stop_event: threading.Event,
    on_progress: Callable[[dict], None],
    model_ref: str,
) -> None:
    last_bytes = _directory_size(repo_dir)
    last_ts = time.monotonic()
    while not stop_event.wait(1.0):
        now = time.monotonic()
        current_bytes = _directory_size(repo_dir)
        elapsed = max(1e-6, now - last_ts)
        speed_bps = max(0.0, (current_bytes - last_bytes) / elapsed)
        last_bytes = current_bytes
        last_ts = now
        on_progress(
            {
                "message": f"Downloading {model_ref}...",
                "downloadedBytes": current_bytes,
                "speedBps": speed_bps,
            }
        )


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def _friendly_download_error(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    if any(token in lowered for token in ("401", "403", "gated", "private", "not authorized", "access token")):
        return "Private or gated Hugging Face models are not supported. Use a public faster-whisper/CTranslate2 model."
    if "repo id" in lowered or "repo_id" in lowered:
        return f"{type(error).__name__}: {text}"
    if "repository not found" in lowered or "not found" in lowered:
        return "Hugging Face model was not found or is not public. Private models are not supported."
    return f"{type(error).__name__}: {text}"


def _present_model_files(path: Path) -> list[str]:
    names = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocabulary.json",
        "vocab.json",
        "merges.txt",
        "model.bin",
        "model.safetensors",
        "README.md",
    ]
    present = [name for name in names if (path / name).exists()]
    present.extend(sorted(item.name for item in path.glob("model-*.safetensors")))
    present.extend(sorted(item.name for item in path.glob("pytorch_model-*.bin")))
    return present


def _weight_file_records(path: Path) -> list[dict]:
    records: list[dict] = []
    candidates = [path / "model.bin", path / "model.safetensors", path / "pytorch_model.bin"]
    candidates.extend(sorted(path.glob("model-*.safetensors")))
    candidates.extend(sorted(path.glob("pytorch_model-*.bin")))
    for candidate in candidates:
        try:
            if candidate.exists():
                records.append({"name": candidate.name, "bytes": int(candidate.stat().st_size)})
        except OSError:
            pass
    return records


def _read_json_summary(path: Path, keys: list[str]) -> dict:
    try:
        if not path.exists() or path.stat().st_size > 5_000_000:
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    summary: dict = {}
    for key in keys:
        if key in data:
            value = data[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = value
            elif isinstance(value, list):
                summary[key] = value[:12]
            elif isinstance(value, dict):
                summary[key] = {str(k): value[k] for k in list(value.keys())[:12]}
    return summary


def _read_model_card_metadata(path: Path) -> dict:
    readme = _best_model_dir(path) / "README.md"
    if not readme.exists():
        readme = path / "README.md"
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    first_heading = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            first_heading = stripped.lstrip("#").strip()
            break

    metadata: dict = {}
    lines = text.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:80]:
            stripped = line.strip()
            if stripped == "---":
                break
            if ":" not in stripped or stripped.startswith("-"):
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            if key in {"license", "language", "pipeline_tag", "library_name", "tags", "base_model"}:
                metadata[key] = value.strip().strip("'\"")
    if first_heading:
        metadata["title"] = first_heading
    return metadata


@contextmanager
def _temporary_proxy_env(proxy: str) -> Iterator[None]:
    with _DOWNLOAD_ENV_LOCK:
        proxy_text = str(proxy or "").strip()
        if not proxy_text:
            yield
            return

        keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ[key] = proxy_text
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _hf_repo_dir(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / f"models--{repo_id.replace('/', '--')}"


def _resolve_model_root(model_name: str, *, models_dir: str | Path | None = None) -> Path:
    model_ref = normalize_model_reference(model_name)
    direct_path = Path(model_ref).expanduser()
    if direct_path.exists():
        return direct_path.resolve()
    root = model_cache_dir(models_dir)
    if _is_repo_id(model_ref):
        for cache_root in _candidate_cache_roots(root):
            candidate = _hf_repo_dir(cache_root, model_ref)
            if _path_exists(candidate):
                return candidate
    return _hf_repo_dir(root, model_ref) if _is_repo_id(model_ref) else direct_path.resolve()


def _candidate_cache_roots(root: Path) -> list[Path]:
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


def _scan_one_models_root(root: Path, *, strict: bool) -> list[dict]:
    try:
        children = sorted(root.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        if strict:
            raise
        return [_inaccessible_model_record(root, exc)]

    records: list[dict] = []
    for child in children:
        try:
            is_dir = child.is_dir()
        except OSError as exc:
            records.append(_inaccessible_model_record(child, exc))
            continue
        if not is_dir:
            continue
        try:
            if child.name.startswith("models--"):
                repo_id = _repo_id_from_cache_folder(child.name)
                if not repo_id:
                    continue
                record = _local_model_record(repo_id, child, source="huggingface-cache")
            elif _looks_like_model_folder(child):
                record = _local_model_record(str(child), child, source="local-folder", label=child.name)
            else:
                continue
        except OSError as exc:
            record = _inaccessible_model_record(child, exc)
        records.append(record)
    return records


def _path_exists(path: Path) -> bool:
    try:
        return Path(path).exists()
    except OSError:
        return True


def _repo_id_from_cache_folder(name: str) -> str:
    raw = str(name or "")
    if not raw.startswith("models--"):
        return ""
    parts = [part.strip() for part in raw[len("models--"):].split("--") if part.strip()]
    if len(parts) < 2:
        return ""
    return f"{parts[0]}/{'--'.join(parts[1:])}"


def _best_model_dir(path: Path) -> Path:
    if (path / "snapshots").is_dir():
        snapshots = [item for item in (path / "snapshots").iterdir() if item.is_dir()]
        compatible = [item for item in snapshots if (item / "config.json").exists() and _has_ctranslate2_weight_file(item)]
        if compatible:
            return max(compatible, key=lambda item: item.stat().st_mtime)
        if snapshots:
            return max(snapshots, key=lambda item: item.stat().st_mtime)
    return path


def _has_ctranslate2_weight_file(path: Path) -> bool:
    min_size = 1_000_000
    candidate = path / "model.bin"
    try:
        return candidate.exists() and candidate.stat().st_size > min_size
    except Exception:
        return False


def _has_transformers_weight_file(path: Path) -> bool:
    min_size = 1_000_000
    for pattern in ("model.safetensors", "model-*.safetensors", "pytorch_model.bin", "pytorch_model-*.bin"):
        for candidate in path.glob(pattern):
            try:
                if candidate.exists() and candidate.stat().st_size > min_size:
                    return True
            except Exception:
                pass
    return False


def _has_tokenizer_file(path: Path) -> bool:
    return any((path / name).exists() for name in ("tokenizer.json", "vocabulary.json", "vocabulary.txt", "vocab.json"))


def _looks_like_model_folder(path: Path) -> bool:
    if (path / "snapshots").is_dir() or (path / "config.json").exists():
        return True
    return any((path / name).exists() for name in ("model.bin", "model.safetensors", "pytorch_model.bin"))


def _local_model_record(name: str, path: Path, *, source: str, label: str = "") -> dict:
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


def _inaccessible_model_record(path: Path, error: Exception) -> dict:
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


def _is_repo_id(value: str) -> bool:
    text = str(value or "").strip()
    return is_valid_huggingface_repo_id(text) and not Path(text).expanduser().exists()
