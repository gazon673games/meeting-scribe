from __future__ import annotations

from pathlib import Path

from application.model_download_parts.catalog import delete_local_model, scan_local_models
from application.model_download_parts.constants import BUILTIN_MODELS
from application.model_download_parts.download import (
    download_model_async,
    friendly_download_error as _friendly_download_error,
    monitor_download_progress as _monitor_download_progress,
    temporary_proxy_env as _temporary_proxy_env,
)
from application.model_download_parts.filesystem import directory_size as _directory_size
from application.model_download_parts.inspection import (
    best_model_dir as _best_model_dir,
    coerce_summary_value as _coerce_summary_value,
    has_ctranslate2_weight_file as _has_ctranslate2_weight_file,
    has_tokenizer_file as _has_tokenizer_file,
    has_transformers_weight_file as _has_transformers_weight_file,
    inspect_model_path,
    looks_like_model_folder as _looks_like_model_folder,
    model_metadata,
    parse_readme_frontmatter as _parse_readme_frontmatter,
    present_model_files as _present_model_files,
    raw_path_exists as _path_exists,
    read_json_summary as _read_json_summary,
    read_model_card_metadata as _read_model_card_metadata,
    weight_file_records as _weight_file_records,
)
from application.model_download_parts.paths import (
    candidate_cache_roots as _candidate_cache_roots,
    hf_repo_dir as _hf_repo_dir,
    model_cache_dir,
    repo_id_from_cache_folder as _repo_id_from_cache_folder,
    resolve_model_root as _resolve_model_root,
)
from application.model_download_parts.references import (
    is_builtin_model,
    is_repo_id as _is_repo_id,
    is_valid_huggingface_repo_id,
    normalize_model_reference,
)

__all__ = [
    "BUILTIN_MODELS",
    "delete_local_model",
    "download_model_async",
    "inspect_model_path",
    "is_builtin_model",
    "is_model_cached",
    "is_valid_huggingface_repo_id",
    "model_cache_dir",
    "model_metadata",
    "normalize_model_reference",
    "scan_local_models",
    "_temporary_proxy_env",
]


def is_model_cached(model_name: str, *, models_dir: str | Path | None = None) -> bool:
    model_ref = normalize_model_reference(model_name)
    if not model_ref:
        return False
    try:
        return bool(inspect_model_path(_resolve_model_root(model_ref, models_dir=models_dir))["compatible"])
    except Exception:
        return False
