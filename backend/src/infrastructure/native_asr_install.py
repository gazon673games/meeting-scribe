from __future__ import annotations

import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from application.native_asr_models import (
    is_native_asr_model_cached,
    native_asr_archive_url,
    native_asr_model_dir,
    native_asr_required_files,
)


def install_native_asr_model(
    model_name: str,
    *,
    models_dir: str | Path,
    proxy: str = '',
    on_progress: Callable[[dict], None] | None = None,
) -> Path:
    url = native_asr_archive_url(model_name)
    if url is None:
        raise ValueError(f'No archive installer for {model_name}')
    target = native_asr_model_dir(model_name, models_dir=models_dir)
    if is_native_asr_model_cached(model_name, models_dir=models_dir):
        return target
    if target.exists():
        raise FileExistsError(
            f'Incomplete model directory already exists: {target}'
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix='.native-asr-', dir=target.parent
    ) as raw:
        staging = Path(raw)
        archive_path = staging / 'model.tar.bz2'
        _download_archive(url, archive_path, proxy, on_progress)
        extracted = staging / target.name
        extract_native_asr_model(model_name, archive_path, extracted)
        extracted.rename(target)
    return target


def _download_archive(
    url: str,
    destination: Path,
    proxy: str,
    on_progress: Callable[[dict], None] | None,
) -> None:
    proxies = {'http': proxy, 'https': proxy} if proxy else {}
    opener = build_opener(ProxyHandler(proxies))
    downloaded_bytes = 0
    started_at = time.monotonic()
    last_report_at = started_at
    with (
        opener.open(url, timeout=60) as response,
        destination.open('wb') as output,
    ):
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded_bytes += len(chunk)
            now = time.monotonic()
            if on_progress is not None and now - last_report_at >= 0.25:
                last_report_at = now
                elapsed_s = max(0.001, now - started_at)
                on_progress(
                    {
                        'message': 'Downloading ONNX model...',
                        'downloadedBytes': downloaded_bytes,
                        'speedBps': downloaded_bytes / elapsed_s,
                    }
                )
    if on_progress is not None:
        on_progress(
            {
                'message': 'Validating ONNX model...',
                'downloadedBytes': downloaded_bytes,
                'speedBps': 0,
            }
        )


def extract_native_asr_model(
    model_name: str, archive_path: Path, destination: Path
) -> None:
    directory = native_asr_model_dir(model_name).name
    required = native_asr_required_files(model_name)
    with tarfile.open(archive_path, mode='r:bz2') as archive:
        for name in required:
            member = archive.getmember(f'{directory}/{name}')
            if not member.isfile() or member.size <= 0:
                raise ValueError(f'Invalid model archive member: {member.name}')
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f'Missing model archive member: {member.name}')
            output = destination / name
            output.parent.mkdir(parents=True, exist_ok=True)
            with source, output.open('wb') as target:
                shutil.copyfileobj(source, target)
