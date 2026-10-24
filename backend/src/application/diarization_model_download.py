from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request

from infrastructure.background_tasks import model_download_tasks
from infrastructure.http_transport import open_url


@dataclass(frozen=True)
class DiarizationModelSpec:
    name: str
    label: str
    file_name: str
    url: str
    backend: str = 'sherpa_onnx'
    provider: str = 'cpu'


RECOMMENDED_DIARIZATION_MODELS = [
    DiarizationModelSpec(
        name='sherpa-onnx-3dspeaker-eres2net-common',
        label='Sherpa-ONNX 3D-Speaker ERes2Net',
        file_name='3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx',
        url=(
            'https://github.com/k2-fsa/sherpa-onnx/releases/download/'
            'speaker-recongition-models/3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx'
        ),
    ),
]


def diarization_models_dir(
    project_root: Path,
    models_dir: str | Path | None = None,
    *,
    subdir: str = 'diarization',
) -> Path:
    root = (
        Path(models_dir).expanduser().resolve()
        if models_dir
        else Path(project_root).resolve() / 'models'
    )
    return root / subdir


def list_diarization_models(
    *,
    project_root: Path,
    models_dir: str | Path | None = None,
    downloads: dict[str, dict] | None = None,
    subdir: str = 'diarization',
) -> dict:
    root = diarization_models_dir(project_root, models_dir, subdir=subdir)
    root.mkdir(parents=True, exist_ok=True)
    active = downloads or {}
    models = [
        _recommended_record(spec, root, active)
        for spec in RECOMMENDED_DIARIZATION_MODELS
    ]
    known_paths = {str(record.get('path') or '') for record in models}
    for path in sorted(root.glob('*.onnx'), key=lambda item: item.name.lower()):
        if str(path) in known_paths:
            continue
        models.append(_local_record(path))
    return {'models': models, 'modelsDir': str(root)}


def default_cached_diarization_model(
    *,
    project_root: Path,
    models_dir: str | Path | None = None,
    subdir: str = 'diarization',
) -> dict | None:
    root = diarization_models_dir(project_root, models_dir, subdir=subdir)
    for spec in RECOMMENDED_DIARIZATION_MODELS:
        path = root / spec.file_name
        if path.exists() and _file_size(path) > 0:
            return _recommended_record(spec, root, {})
    return None


def download_diarization_model_async(
    *,
    name: str,
    project_root: Path,
    on_progress: Callable[[dict], None],
    on_done: Callable[[str | None], None],
    models_dir: str | Path | None = None,
    proxy: str = '',
    subdir: str = 'diarization',
) -> None:
    spec = _model_spec(name)

    def _run() -> None:
        completion_error = None
        try:
            root = diarization_models_dir(
                project_root, models_dir, subdir=subdir
            )
            root.mkdir(parents=True, exist_ok=True)
            target = root / spec.file_name
            tmp_target = target.with_suffix(target.suffix + '.part')
            _download_file(
                url=spec.url,
                target=tmp_target,
                proxy=proxy,
                on_progress=lambda payload: on_progress(
                    {'name': spec.name, 'path': str(target), **payload}
                ),
            )
            if tmp_target.stat().st_size <= 0:
                raise RuntimeError('Downloaded model is empty')
            tmp_target.replace(target)
            on_progress(
                {
                    'name': spec.name,
                    'message': 'Downloaded',
                    'downloadedBytes': _file_size(target),
                    'speedBps': 0,
                    'path': str(target),
                }
            )
        except Exception as error:
            completion_error = f'{type(error).__name__}: {error}'
        on_done(completion_error)

    model_download_tasks.start(target=_run, name='diarization-model-download')


def delete_diarization_model(
    *,
    project_root: Path,
    path: str,
    models_dir: str | Path | None = None,
    subdir: str = 'diarization',
) -> None:
    root = diarization_models_dir(
        project_root, models_dir, subdir=subdir
    ).resolve()
    target = Path(path).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(
            'Can only delete Speaker ID models inside the configured Speaker ID models directory'
        ) from error
    if target.exists():
        target.unlink()


def _download_file(
    *,
    url: str,
    target: Path,
    proxy: str,
    on_progress: Callable[[dict], None],
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={'User-Agent': 'MeetingScribe/1.0'})
    downloaded = 0
    last_bytes = 0
    last_ts = time.monotonic()
    with open_url(request, proxy=proxy, timeout=30) as response:
        total = int(response.headers.get('Content-Length') or 0)
        with target.open('wb') as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                elapsed = max(1e-6, now - last_ts)
                speed_bps = max(0.0, (downloaded - last_bytes) / elapsed)
                last_ts = now
                last_bytes = downloaded
                on_progress(
                    {
                        'message': 'Downloading...',
                        'downloadedBytes': downloaded,
                        'totalBytes': total,
                        'speedBps': speed_bps,
                    }
                )


def _recommended_record(
    spec: DiarizationModelSpec, root: Path, downloads: dict[str, dict]
) -> dict:
    path = root / spec.file_name
    download = downloads.get(spec.name) or {}
    cached = path.exists() and _file_size(path) > 0
    return {
        'name': spec.name,
        'label': spec.label,
        'fileName': spec.file_name,
        'backend': spec.backend,
        'provider': spec.provider,
        'url': spec.url,
        'path': str(path),
        'cached': cached,
        'compatible': cached,
        'status': 'ready' if cached else 'downloadable',
        'source': 'recommended',
        'recommended': True,
        'downloadable': True,
        'deletable': cached,
        'bytes': _file_size(path),
        **_download_fields(download),
    }


def _local_record(path: Path) -> dict:
    return {
        'name': str(path),
        'label': path.name,
        'fileName': path.name,
        'backend': 'sherpa_onnx',
        'provider': 'cpu',
        'url': '',
        'path': str(path),
        'cached': True,
        'compatible': True,
        'status': 'ready',
        'source': 'local-file',
        'recommended': False,
        'downloadable': False,
        'deletable': True,
        'bytes': _file_size(path),
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


def _model_spec(name: str) -> DiarizationModelSpec:
    text = str(name or '').strip()
    for spec in RECOMMENDED_DIARIZATION_MODELS:
        if text in {spec.name, spec.file_name, spec.url}:
            return spec
    raise ValueError(f'Unknown Speaker ID model: {name}')


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size) if path.exists() else 0
    except OSError:
        return 0
