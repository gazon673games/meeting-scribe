from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from application.llm_model_sources import (
    PREFERRED_GGUF_QUANTS as PREFERRED_GGUF_QUANTS,
)
from application.llm_model_sources import LlmSource as LlmSource
from application.llm_model_sources import _choose_gguf_file as _choose_gguf_file
from application.llm_model_sources import (
    _huggingface_url_source as _huggingface_url_source,
)
from application.llm_model_sources import (
    _repo_and_filename as _repo_and_filename,
)
from application.llm_model_sources import _safe_filename as _safe_filename
from application.llm_model_sources import _safe_folder as _safe_folder
from application.llm_model_sources import parse_llm_source as parse_llm_source
from infrastructure.background_tasks import model_download_tasks
from infrastructure.llm_model_transfer import (
    _download_hf_file as _download_hf_file,
)
from infrastructure.llm_model_transfer import _download_url as _download_url
from infrastructure.llm_model_transfer import _hf_headers as _hf_headers
from infrastructure.llm_model_transfer import (
    _resolve_repo_source as _resolve_repo_source,
)


def llm_models_dir(
    project_root: Path, models_dir: str | Path | None = None
) -> Path:
    if models_dir:
        return Path(models_dir).expanduser().resolve()
    return Path(project_root).resolve() / 'models' / 'llm'


def list_llm_models(
    *,
    project_root: Path,
    models_dir: str | Path | None = None,
    downloads: dict[str, dict] | None = None,
) -> dict:
    root = llm_models_dir(project_root, models_dir)
    root.mkdir(parents=True, exist_ok=True)
    active = downloads or {}
    models = [
        _local_record(path, active)
        for path in sorted(
            root.rglob('*.gguf'), key=lambda item: item.name.lower()
        )
    ]

    known = set()
    for model in models:
        known.add(str(model.get('name') or ''))
        known.add(str(model.get('fileName') or ''))
        model_path = str(model.get('path') or '')
        if model_path:
            known.add(Path(model_path).parent.name)
    for name, record in active.items():
        if str(name) in known:
            continue
        models.append(_download_record(str(name), record))
    return {'models': models, 'modelsDir': str(root)}


def download_llm_model_async(
    *,
    name: str,
    project_root: Path,
    on_progress: Callable[[dict], None],
    on_done: Callable[[str | None], None],
    models_dir: str | Path | None = None,
    proxy: str = '',
) -> None:
    source = parse_llm_source(name)

    def _run() -> None:
        completion_error = None
        try:
            resolved_source = _resolve_repo_source(source, proxy=proxy)
            root = llm_models_dir(project_root, models_dir)
            root.mkdir(parents=True, exist_ok=True)
            target_dir = root / resolved_source.folder
            target = (target_dir / resolved_source.filename).resolve()
            try:
                target.relative_to(root.resolve())
                target.relative_to(target_dir.resolve())
            except ValueError as error:
                raise ValueError(
                    'Language model file must stay inside its model directory'
                ) from error
            target_dir.mkdir(parents=True, exist_ok=True)

            on_progress(
                {
                    'message': f'Downloading {resolved_source.filename}...',
                    'path': str(target),
                    'downloadedBytes': _dir_size(target_dir),
                    'speedBps': 0,
                }
            )
            if resolved_source.repo_id:
                _download_hf_file(
                    resolved_source,
                    target,
                    proxy=proxy,
                    on_progress=on_progress,
                )
            else:
                _download_url(
                    resolved_source.url,
                    target,
                    proxy=proxy,
                    on_progress=on_progress,
                )

            if not target.exists() or target.stat().st_size <= 0:
                raise RuntimeError('Downloaded GGUF file is empty')
            on_progress(
                {
                    'message': 'Downloaded',
                    'path': str(target),
                    'downloadedBytes': _file_size(target),
                    'speedBps': 0,
                }
            )
        except Exception as error:
            completion_error = f'{type(error).__name__}: {error}'
        on_done(completion_error)

    model_download_tasks.start(target=_run, name='llm-model-download')


def delete_llm_model(
    *, project_root: Path, path: str, models_dir: str | Path | None = None
) -> None:
    root = llm_models_dir(project_root, models_dir).resolve()
    target = Path(path).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(
            'Can only delete language models inside the configured language models directory'
        ) from error
    if target.exists():
        target.unlink()
    parent = target.parent
    if parent != root and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


def _local_record(path: Path, downloads: dict[str, dict]) -> dict:
    name = path.stem
    download = downloads.get(name) or downloads.get(path.name) or {}
    return {
        'name': name,
        'label': path.name,
        'fileName': path.name,
        'path': str(path),
        'modelAlias': name,
        'cached': True,
        'compatible': True,
        'status': 'ready',
        'source': 'local-file',
        'downloadable': False,
        'deletable': True,
        'bytes': _file_size(path),
        **_download_fields(download),
    }


def _download_record(name: str, download: dict) -> dict:
    return {
        'name': name,
        'label': name,
        'fileName': name,
        'path': str(download.get('path') or ''),
        'modelAlias': Path(name).stem,
        'cached': False,
        'compatible': False,
        'status': 'downloading'
        if download.get('state') == 'downloading'
        else 'error',
        'source': 'download',
        'downloadable': False,
        'deletable': False,
        'bytes': 0,
        **_download_fields(download),
    }


def _download_fields(download: dict) -> dict:
    return {
        'downloading': download.get('state') == 'downloading',
        'downloadDone': download.get('state') == 'done',
        'downloadError': str(download.get('error') or ''),
        'downloadMessage': str(download.get('message') or ''),
        'downloadedBytes': int(download.get('downloadedBytes') or 0),
        'totalBytes': int(download.get('totalBytes') or 0),
        'speedBps': float(download.get('speedBps') or 0.0),
        'downloadUsesProxy': bool(download.get('proxy', False)),
    }


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size) if path.exists() else 0
    except OSError:
        return 0


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(_file_size(item) for item in path.rglob('*') if item.is_file())
