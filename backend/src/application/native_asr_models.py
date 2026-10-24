from __future__ import annotations

import os
from pathlib import Path

NEMOTRON_MODEL_ID = 'sherpa-nemotron-3.5-asr-streaming-0.6b'
NEMOTRON_MODEL_DIR_NAME = (
    'sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11'
)
NEMOTRON_REQUIRED_FILES = (
    'encoder.int8.onnx',
    'decoder.int8.onnx',
    'joiner.int8.onnx',
    'tokens.txt',
)
GIGAAM_MODEL_ID = 'gigaam-v3-e2e-rnnt'
GIGAAM_MODEL_DIR_NAME = 'gigaam-v3-e2e-rnnt'
GIGAAM_REQUIRED_FILES = (
    'config.json',
    'modeling_gigaam.py',
    'pytorch_model.bin',
    'tokenizer.model',
)
PARAKEET_MODEL_ID = 'sherpa-parakeet-tdt-0.6b-v3'
PARAKEET_MODEL_DIR_NAME = 'sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8'
QWEN_ASR_MODEL_ID = 'sherpa-qwen3-asr-0.6b'
QWEN_ASR_MODEL_DIR_NAME = 'sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25'
QWEN_ASR_REQUIRED_FILES = (
    'conv_frontend.onnx',
    'encoder.int8.onnx',
    'decoder.int8.onnx',
    'tokenizer/vocab.json',
    'tokenizer/merges.txt',
    'tokenizer/tokenizer_config.json',
)
SHERPA_OFFLINE_MODEL_IDS = frozenset({PARAKEET_MODEL_ID, QWEN_ASR_MODEL_ID})

_MODEL_LAYOUTS = {
    NEMOTRON_MODEL_ID: (NEMOTRON_MODEL_DIR_NAME, NEMOTRON_REQUIRED_FILES),
    GIGAAM_MODEL_ID: (GIGAAM_MODEL_DIR_NAME, GIGAAM_REQUIRED_FILES),
    PARAKEET_MODEL_ID: (PARAKEET_MODEL_DIR_NAME, NEMOTRON_REQUIRED_FILES),
    QWEN_ASR_MODEL_ID: (QWEN_ASR_MODEL_DIR_NAME, QWEN_ASR_REQUIRED_FILES),
}


def native_asr_required_files(model_name: str) -> tuple[str, ...]:
    return _MODEL_LAYOUTS[model_name][1]


def native_asr_archive_url(model_name: str) -> str | None:
    if model_name not in {NEMOTRON_MODEL_ID, *SHERPA_OFFLINE_MODEL_IDS}:
        return None
    directory = _MODEL_LAYOUTS[model_name][0]
    return (
        'https://github.com/k2-fsa/sherpa-onnx/releases/download/'
        f'asr-models/{directory}.tar.bz2'
    )


def is_native_asr_model(model_name: str) -> bool:
    return str(model_name or '').strip() in _MODEL_LAYOUTS


def native_asr_model_dir(
    model_name: str,
    *,
    models_dir: str | Path | None = None,
    project_root: Path | None = None,
) -> Path:
    if not is_native_asr_model(model_name):
        raise ValueError(f'Unknown native ASR model: {model_name}')
    model_id = str(model_name or '').strip()
    root = _models_root(models_dir=models_dir, project_root=project_root)
    return root / _MODEL_LAYOUTS[model_id][0]


def is_native_asr_model_cached(
    model_name: str,
    *,
    models_dir: str | Path | None = None,
    project_root: Path | None = None,
) -> bool:
    if not is_native_asr_model(model_name):
        return False
    root = native_asr_model_dir(
        model_name, models_dir=models_dir, project_root=project_root
    )
    required_files = _MODEL_LAYOUTS[str(model_name or '').strip()][1]
    return all(
        (root / filename).is_file() and (root / filename).stat().st_size > 0
        for filename in required_files
    )


def _models_root(
    *, models_dir: str | Path | None, project_root: Path | None
) -> Path:
    if models_dir:
        return Path(models_dir).expanduser().resolve()
    configured = str(os.environ.get('HF_HUB_CACHE') or '').strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if project_root is not None:
        return Path(project_root).resolve() / 'models'
    return Path.cwd().resolve() / 'models'
