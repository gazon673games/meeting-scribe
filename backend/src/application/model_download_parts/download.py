from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

from application.model_download_parts.filesystem import directory_size
from application.model_download_parts.inspection import inspect_model_path
from application.model_download_parts.paths import hf_repo_dir, model_cache_dir
from application.model_download_parts.references import (
    is_valid_huggingface_repo_id,
    normalize_model_reference,
)
from infrastructure.background_tasks import model_download_tasks
from infrastructure.huggingface_transfer import download_snapshot
from infrastructure.huggingface_transfer import (
    temporary_proxy_env as temporary_proxy_env,
)


def download_model_async(
    model_name: str,
    on_progress: Callable[[dict], None],
    on_done: Callable[[str | None], None],
    *,
    models_dir: str | Path | None = None,
    proxy: str = '',
) -> None:
    def _run() -> None:
        stop_event: threading.Event | None = None
        monitor: threading.Thread | None = None
        monitor_started = False
        completion_error = None
        try:
            from application.native_asr_models import is_native_asr_model

            if is_native_asr_model(model_name):
                from infrastructure.native_asr_install import (
                    install_native_asr_model,
                )

                install_native_asr_model(
                    model_name,
                    models_dir=model_cache_dir(models_dir),
                    proxy=proxy,
                    on_progress=on_progress,
                )
                return
            model_ref = normalize_model_reference(model_name)
            if not model_ref:
                raise ValueError('Model name is empty')
            if Path(model_ref).expanduser().exists():
                raise ValueError('Local model folders do not need downloading')
            if not is_valid_huggingface_repo_id(model_ref):
                raise ValueError(f'Invalid Hugging Face repo id: {model_ref}')
            cache_dir = model_cache_dir(models_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            repo_dir = hf_repo_dir(cache_dir, model_ref)
            on_progress(
                {
                    'message': f'Downloading {model_ref}...',
                    'downloadedBytes': directory_size(repo_dir),
                    'speedBps': 0,
                }
            )
            stop_event = threading.Event()
            monitor = threading.Thread(
                target=monitor_download_progress,
                args=(repo_dir, stop_event, on_progress, model_ref),
                name='model-download-progress',
                daemon=True,
            )
            monitor.start()
            monitor_started = True
            download_snapshot(
                repo_id=model_ref,
                cache_dir=str(cache_dir),
                proxy=proxy,
            )
            on_progress(
                {
                    'message': 'Validating downloaded model...',
                    'downloadedBytes': directory_size(repo_dir),
                    'speedBps': 0,
                }
            )
            info = inspect_model_path(repo_dir)
            if not info.get('compatible', False):
                missing = (
                    ', '.join(str(item) for item in info.get('missing', []))
                    or 'compatible CTranslate2 files'
                )
                raise ValueError(
                    f'Downloaded model is not faster-whisper compatible: missing {missing}'
                )
            on_progress(
                {
                    'message': 'Downloaded',
                    'downloadedBytes': directory_size(repo_dir),
                    'speedBps': 0,
                }
            )
        except Exception as error:
            completion_error = friendly_download_error(error)
        finally:
            if stop_event is not None:
                stop_event.set()
            if monitor is not None and monitor_started:
                monitor.join()
            on_done(completion_error)

    model_download_tasks.start(target=_run, name='model-download')


def monitor_download_progress(
    repo_dir: Path,
    stop_event: threading.Event,
    on_progress: Callable[[dict], None],
    model_ref: str,
) -> None:
    last_bytes = directory_size(repo_dir)
    last_ts = time.monotonic()
    while not stop_event.wait(1.0):
        now = time.monotonic()
        current_bytes = directory_size(repo_dir)
        elapsed = max(1e-6, now - last_ts)
        speed_bps = max(0.0, (current_bytes - last_bytes) / elapsed)
        last_bytes = current_bytes
        last_ts = now
        on_progress(
            {
                'message': f'Downloading {model_ref}...',
                'downloadedBytes': current_bytes,
                'speedBps': speed_bps,
            }
        )


def friendly_download_error(error: Exception) -> str:
    text = str(error)
    lowered = text.lower()
    if any(
        token in lowered
        for token in (
            '401',
            '403',
            'gated',
            'private',
            'not authorized',
            'access token',
        )
    ):
        return 'Private or gated Hugging Face models are not supported. Use a public faster-whisper/CTranslate2 model.'
    if 'repo id' in lowered or 'repo_id' in lowered:
        return f'{type(error).__name__}: {text}'
    if 'repository not found' in lowered or 'not found' in lowered:
        return 'Hugging Face model was not found or is not public. Private models are not supported.'
    return f'{type(error).__name__}: {text}'
