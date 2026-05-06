from __future__ import annotations

import json
from pathlib import Path

from application.model_download_parts.constants import BUILTIN_MODEL_REPOS, BUILTIN_MODELS, README_KEYS
from application.model_download_parts.filesystem import directory_size
from application.model_download_parts.paths import model_cache_dir, resolve_model_root
from application.model_download_parts.references import is_repo_id, normalize_model_reference


def inspect_model_path(path: str | Path) -> dict:
    raw_path = Path(path).expanduser()
    target = best_model_dir(raw_path)
    missing: list[str] = []
    warnings: list[str] = []
    if not raw_path_exists(raw_path):
        return {"compatible": False, "status": "missing", "path": str(raw_path), "missing": ["folder"]}
    if not (target / "config.json").exists():
        missing.append("config.json")
    if not has_ctranslate2_weight_file(target):
        if has_transformers_weight_file(target):
            missing.append("model.bin")
            warnings.append(
                "Transformers/safetensors weights are not supported by faster-whisper; use a CTranslate2 model.bin export."
            )
        else:
            missing.append("model.bin")
    if not has_tokenizer_file(target):
        missing.append("tokenizer.json or vocabulary.json")
    compatible = not missing
    if compatible:
        status = "compatible"
        model_format = "ctranslate2"
    elif has_transformers_weight_file(target):
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
    model_root = resolve_model_root(model_ref, models_dir=root)
    info = inspect_model_path(model_root)
    resolved = Path(str(info.get("path") or model_root))
    model_root_exists = model_root.exists()
    direct_path = Path(model_ref).expanduser()
    return {
        "name": raw,
        "normalizedName": model_ref,
        "repoId": model_ref if is_repo_id(model_ref) else "",
        "source": "local-folder" if direct_path.exists() else "huggingface-cache",
        "builtin": raw in BUILTIN_MODELS or model_ref in set(BUILTIN_MODEL_REPOS.values()),
        "cacheDir": str(root),
        "cachePath": str(model_root),
        "resolvedPath": str(resolved),
        "cached": bool(info.get("compatible", False)),
        "compatible": bool(info.get("compatible", False)),
        "status": str(info.get("status") or "unknown"),
        "format": str(info.get("format") or "unknown"),
        "missing": list(info.get("missing") or []),
        "warnings": list(info.get("warnings") or []),
        "totalBytes": directory_size(model_root) if model_root_exists else 0,
        "presentFiles": present_model_files(resolved) if resolved.exists() else [],
        "weightFiles": weight_file_records(resolved) if resolved.exists() else [],
        "config": read_json_summary(
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
        "preprocessor": read_json_summary(
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
        "tokenizer": read_json_summary(
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
        "readme": read_model_card_metadata(model_root),
    }


def present_model_files(path: Path) -> list[str]:
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


def weight_file_records(path: Path) -> list[dict]:
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


def read_json_summary(path: Path, keys: list[str]) -> dict:
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
            ok, coerced = coerce_summary_value(data[key])
            if ok:
                summary[key] = coerced
    return summary


def read_model_card_metadata(path: Path) -> dict:
    readme = best_model_dir(path) / "README.md"
    if not readme.exists():
        readme = path / "README.md"
    try:
        text = readme.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    lines = text.splitlines()
    first_heading = next(
        (line.strip().lstrip("#").strip() for line in lines if line.strip().startswith("#")),
        "",
    )
    metadata = parse_readme_frontmatter(lines)
    if first_heading:
        metadata["title"] = first_heading
    return metadata


def parse_readme_frontmatter(lines: list[str]) -> dict:
    metadata: dict = {}
    if not lines or lines[0].strip() != "---":
        return metadata
    for line in lines[1:80]:
        stripped = line.strip()
        if stripped == "---":
            break
        if ":" not in stripped or stripped.startswith("-"):
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        if key in README_KEYS:
            metadata[key] = value.strip().strip("'\"")
    return metadata


def coerce_summary_value(value: object) -> tuple[bool, object]:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True, value
    if isinstance(value, list):
        return True, value[:12]
    if isinstance(value, dict):
        return True, {str(k): value[k] for k in list(value.keys())[:12]}
    return False, None


def raw_path_exists(path: Path) -> bool:
    try:
        return Path(path).exists()
    except OSError:
        return True


def best_model_dir(path: Path) -> Path:
    if (path / "snapshots").is_dir():
        snapshots = [item for item in (path / "snapshots").iterdir() if item.is_dir()]
        compatible = [item for item in snapshots if (item / "config.json").exists() and has_ctranslate2_weight_file(item)]
        if compatible:
            return max(compatible, key=lambda item: item.stat().st_mtime)
        if snapshots:
            return max(snapshots, key=lambda item: item.stat().st_mtime)
    return path


def has_ctranslate2_weight_file(path: Path) -> bool:
    min_size = 1_000_000
    candidate = path / "model.bin"
    try:
        return candidate.exists() and candidate.stat().st_size > min_size
    except Exception:
        return False


def has_transformers_weight_file(path: Path) -> bool:
    min_size = 1_000_000
    for pattern in ("model.safetensors", "model-*.safetensors", "pytorch_model.bin", "pytorch_model-*.bin"):
        for candidate in path.glob(pattern):
            try:
                if candidate.exists() and candidate.stat().st_size > min_size:
                    return True
            except Exception:
                pass
    return False


def has_tokenizer_file(path: Path) -> bool:
    return any((path / name).exists() for name in ("tokenizer.json", "vocabulary.json", "vocabulary.txt", "vocab.json"))


def looks_like_model_folder(path: Path) -> bool:
    if (path / "snapshots").is_dir() or (path / "config.json").exists():
        return True
    return any((path / name).exists() for name in ("model.bin", "model.safetensors", "pytorch_model.bin"))
