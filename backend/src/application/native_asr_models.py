from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


NEMOTRON_MODEL_ID = "sherpa-nemotron-3.5-asr-streaming-0.6b"
NEMOTRON_MODEL_DIR_NAME = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"
NEMOTRON_REQUIRED_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)
GIGAAM_MODEL_ID = "gigaam-v3-e2e-rnnt"
GIGAAM_MODEL_DIR_NAME = "gigaam-v3-e2e-rnnt"
GIGAAM_REQUIRED_FILES = (
    "config.json",
    "modeling_gigaam.py",
    "pytorch_model.bin",
    "tokenizer.model",
)

_MODEL_LAYOUTS = {
    NEMOTRON_MODEL_ID: (NEMOTRON_MODEL_DIR_NAME, NEMOTRON_REQUIRED_FILES),
    GIGAAM_MODEL_ID: (GIGAAM_MODEL_DIR_NAME, GIGAAM_REQUIRED_FILES),
}


def is_native_asr_model(model_name: str) -> bool:
    return str(model_name or "").strip() in _MODEL_LAYOUTS


def native_asr_model_dir(
    model_name: str,
    *,
    models_dir: str | Path | None = None,
    project_root: Optional[Path] = None,
) -> Path:
    if not is_native_asr_model(model_name):
        raise ValueError(f"Unknown native ASR model: {model_name}")
    model_id = str(model_name or "").strip()
    root = _models_root(models_dir=models_dir, project_root=project_root)
    return root / _MODEL_LAYOUTS[model_id][0]


def is_native_asr_model_cached(
    model_name: str,
    *,
    models_dir: str | Path | None = None,
    project_root: Optional[Path] = None,
) -> bool:
    if not is_native_asr_model(model_name):
        return False
    root = native_asr_model_dir(model_name, models_dir=models_dir, project_root=project_root)
    required_files = _MODEL_LAYOUTS[str(model_name or "").strip()][1]
    return all((root / filename).is_file() for filename in required_files)


def _models_root(*, models_dir: str | Path | None, project_root: Optional[Path]) -> Path:
    if models_dir:
        return Path(models_dir).expanduser().resolve()
    configured = str(os.environ.get("HF_HUB_CACHE") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if project_root is not None:
        return Path(project_root).resolve() / "models"
    return Path.cwd().resolve() / "models"
