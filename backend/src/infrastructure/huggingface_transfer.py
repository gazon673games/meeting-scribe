from __future__ import annotations

import inspect
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_DOWNLOAD_ENV_LOCK = threading.RLock()


def download_snapshot(*, repo_id: str, cache_dir: str, proxy: str) -> None:
    from huggingface_hub import snapshot_download

    proxy = str(proxy or '').strip()
    if not proxy:
        # This request must not observe a legacy download's temporary proxy.
        with _DOWNLOAD_ENV_LOCK:
            snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=False,
            )
        return
    if 'proxies' in inspect.signature(snapshot_download).parameters:
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
            proxies={'http': proxy, 'https': proxy},
        )
        return
    # Older/different SDK interfaces expose proxy configuration only via env.
    # Serialize that compatibility path and restore the previous environment.
    with temporary_proxy_env(proxy):
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=False,
        )


@contextmanager
def temporary_proxy_env(proxy: str) -> Iterator[None]:
    with _DOWNLOAD_ENV_LOCK:
        proxy_text = str(proxy or '').strip()
        if not proxy_text:
            yield
            return

        keys = (
            'HTTP_PROXY',
            'HTTPS_PROXY',
            'ALL_PROXY',
            'http_proxy',
            'https_proxy',
            'all_proxy',
        )
        previous = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ[key] = proxy_text
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
