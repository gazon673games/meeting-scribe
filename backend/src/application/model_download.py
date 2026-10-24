from __future__ import annotations

# ruff: noqa: F401
# Legacy helper aliases remain available to existing callers.
from pathlib import Path

from application.model_download_parts.catalog import (
    delete_local_model,
    scan_local_models,
)
from application.model_download_parts.constants import BUILTIN_MODELS
from application.model_download_parts.download import (
    download_model_async,
)
from application.model_download_parts.download import (
    friendly_download_error as _friendly_download_error,
)
from application.model_download_parts.download import (
    monitor_download_progress as _monitor_download_progress,
)
from application.model_download_parts.download import (
    temporary_proxy_env as _temporary_proxy_env,
)
from application.model_download_parts.filesystem import (
    directory_size as _directory_size,
)
from application.model_download_parts.inspection import (
    best_model_dir as _best_model_dir,
)
from application.model_download_parts.inspection import (
    coerce_summary_value as _coerce_summary_value,
)
from application.model_download_parts.inspection import (
    has_ctranslate2_weight_file as _has_ctranslate2_weight_file,
)
from application.model_download_parts.inspection import (
    has_tokenizer_file as _has_tokenizer_file,
)
from application.model_download_parts.inspection import (
    has_transformers_weight_file as _has_transformers_weight_file,
)
from application.model_download_parts.inspection import (
    inspect_model_path,
    model_metadata,
)
from application.model_download_parts.inspection import (
    looks_like_model_folder as _looks_like_model_folder,
)
from application.model_download_parts.inspection import (
    parse_readme_frontmatter as _parse_readme_frontmatter,
)
from application.model_download_parts.inspection import (
    present_model_files as _present_model_files,
)
from application.model_download_parts.inspection import (
    raw_path_exists as _path_exists,
)
from application.model_download_parts.inspection import (
    read_json_summary as _read_json_summary,
)
from application.model_download_parts.inspection import (
    read_model_card_metadata as _read_model_card_metadata,
)
from application.model_download_parts.inspection import (
    weight_file_records as _weight_file_records,
)
from application.model_download_parts.paths import (
    candidate_cache_roots as _candidate_cache_roots,
)
from application.model_download_parts.paths import (
    hf_repo_dir as _hf_repo_dir,
)
from application.model_download_parts.paths import (
    model_cache_dir,
)
from application.model_download_parts.paths import (
    repo_id_from_cache_folder as _repo_id_from_cache_folder,
)
from application.model_download_parts.paths import (
    resolve_model_root as _resolve_model_root,
)
from application.model_download_parts.references import (
    is_builtin_model,
    is_valid_huggingface_repo_id,
    normalize_model_reference,
)
from application.model_download_parts.references import (
    is_repo_id as _is_repo_id,
)

__all__ = [
    'BUILTIN_MODELS',
    'delete_local_model',
    'download_model_async',
    'inspect_model_path',
    'is_builtin_model',
    'is_model_cached',
    'is_valid_huggingface_repo_id',
    'model_cache_dir',
    'model_metadata',
    'normalize_model_reference',
    'scan_local_models',
    '_temporary_proxy_env',
]


def is_model_cached(
    model_name: str, *, models_dir: str | Path | None = None
) -> bool:
    from application.native_asr_models import (
        is_native_asr_model,
        is_native_asr_model_cached,
    )

    if is_native_asr_model(model_name):
        return is_native_asr_model_cached(model_name, models_dir=models_dir)
    model_ref = normalize_model_reference(model_name)
    if not model_ref:
        return False
    try:
        return bool(
            inspect_model_path(
                _resolve_model_root(model_ref, models_dir=models_dir)
            )['compatible']
        )
    except (OSError, ValueError):
        return False
