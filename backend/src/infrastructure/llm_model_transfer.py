from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request

from application.llm_model_sources import LlmSource, _choose_gguf_file
from infrastructure.http_transport import open_url as _open_model_url


def repository_files(repo_id: str, *, proxy: str) -> list[str]:
    from huggingface_hub import constants
    from huggingface_hub.errors import OfflineModeIsEnabled
    from huggingface_hub.utils import build_hf_headers

    if constants.HF_HUB_OFFLINE:
        raise OfflineModeIsEnabled('Hugging Face offline mode is enabled')
    endpoint = (
        os.environ.get('HF_ENDPOINT') or 'https://huggingface.co'
    ).rstrip('/')
    request = Request(
        f'{endpoint}/api/models/{quote(repo_id, safe="/")}',
        headers=build_hf_headers(user_agent='MeetingScribe/1.0'),
    )
    with _open_model_url(request, proxy=proxy, timeout=30) as response:
        metadata = json.load(response)
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get('siblings'), list
    ):
        raise ValueError('Invalid Hugging Face repository metadata')
    return [
        str(item['rfilename']).strip()
        for item in metadata['siblings']
        if isinstance(item, dict) and isinstance(item.get('rfilename'), str)
    ]


def _resolve_repo_source(source: LlmSource, *, proxy: str) -> LlmSource:
    if not source.repo_id or source.filename:
        return source
    filename = _choose_gguf_file(repository_files(source.repo_id, proxy=proxy))
    return LlmSource(
        filename=filename, folder=source.folder, repo_id=source.repo_id
    )


def _download_hf_file(
    source: LlmSource,
    target: Path,
    *,
    proxy: str,
    on_progress: Callable[[dict], None],
) -> None:
    from huggingface_hub import hf_hub_url  # type: ignore

    url = hf_hub_url(repo_id=source.repo_id, filename=source.filename)
    _download_url(
        url, target, proxy=proxy, on_progress=on_progress, headers=_hf_headers()
    )


def _hf_headers() -> dict[str, str]:
    token = (
        os.environ.get('HF_TOKEN')
        or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    )
    if not token:
        return {}
    return {'Authorization': f'Bearer {token}'}


def _download_url(
    url: str,
    target: Path,
    *,
    proxy: str,
    on_progress: Callable[[dict], None],
    headers: dict[str, str] | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url, headers={'User-Agent': 'MeetingScribe/1.0', **(headers or {})}
    )
    downloaded = 0
    last_bytes = 0
    last_ts = time.monotonic()
    tmp_target = target.with_suffix(target.suffix + '.part')
    with _open_model_url(request, proxy=proxy, timeout=30) as response:
        total = int(response.headers.get('Content-Length') or 0)
        with tmp_target.open('wb') as out:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                speed = max(
                    0.0,
                    (downloaded - last_bytes) / max(1e-6, now - last_ts),
                )
                last_bytes = downloaded
                last_ts = now
                on_progress(
                    {
                        'message': 'Downloading...',
                        'path': str(target),
                        'downloadedBytes': downloaded,
                        'totalBytes': total,
                        'speedBps': speed,
                    }
                )
    tmp_target.replace(target)
