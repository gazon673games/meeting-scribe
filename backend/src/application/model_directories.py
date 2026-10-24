from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from shared.domain.mappings import first_param

ModelKind = Literal['asr', 'diarization', 'llm']


def model_directory(
    project_root: Path,
    config: Mapping,
    kind: ModelKind,
    params: Mapping | None = None,
) -> Path:
    models = config.get('models')
    settings = models if isinstance(models, Mapping) else {}
    overrides = params if params is not None else {}
    raw_base = str(
        first_param(
            overrides,
            'modelsDir',
            'models_dir',
            default=settings.get('cache_dir', ''),
        )
        or ''
    ).strip()
    base = (
        Path(raw_base).expanduser().resolve()
        if raw_base
        else Path(project_root).resolve() / 'models'
    )
    raw_subdir = str(
        first_param(
            overrides,
            'modelsSubdir',
            'models_subdir',
            default=settings.get(f'{kind}_subdir', ''),
        )
        or ''
    ).strip()
    if not raw_subdir:
        if kind == 'diarization':
            return base / 'diarization'
        if kind == 'llm' and not raw_base:
            return base / 'llm'
        return base
    subdir = Path(raw_subdir)
    if subdir.is_absolute() or subdir.drive or subdir.root:
        raise ValueError('Model subdirectory must be relative')
    target = (base / subdir).resolve()
    if not target.is_relative_to(base):
        raise ValueError(
            'Model subdirectory must stay inside the models folder'
        )
    return target
