from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DiarizationModelSpec:
    name: str
    label: str
    file_name: str
    url: str
    backend: str = "sherpa_onnx"
    provider: str = "cpu"


RECOMMENDED_DIARIZATION_MODELS = [
    DiarizationModelSpec(
        name="sherpa-onnx-3dspeaker-eres2net-common",
        label="Sherpa-ONNX 3D-Speaker ERes2Net",
        file_name="3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-recongition-models/3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx"
        ),
    ),
]


def diarization_models_dir(project_root: Path, models_dir: str | Path | None = None) -> Path:
    root = Path(models_dir).expanduser().resolve() if models_dir else Path(project_root).resolve() / "models"
    return root / "diarization"


def list_diarization_models(
    *,
    project_root: Path,
    models_dir: str | Path | None = None,
    downloads: Optional[dict[str, dict]] = None,
) -> dict:
    root = diarization_models_dir(project_root, models_dir)
    root.mkdir(parents=True, exist_ok=True)
    active = downloads or {}
    models = [_recommended_record(spec, root, active) for spec in RECOMMENDED_DIARIZATION_MODELS]
    known_paths = {str(record.get("path") or "") for record in models}
    for path in sorted(root.glob("*.onnx"), key=lambda item: item.name.lower()):
        if str(path) in known_paths:
            continue
        models.append(_local_record(path))
    return {"models": models, "modelsDir": str(root)}


def default_cached_diarization_model(
    *,
    project_root: Path,
    models_dir: str | Path | None = None,
) -> dict | None:
    root = diarization_models_dir(project_root, models_dir)
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
    on_done: Callable[[Optional[str]], None],
    models_dir: str | Path | None = None,
    proxy: str = "",
) -> None:
    spec = _model_spec(name)

    def _run() -> None:
        try:
            root = diarization_models_dir(project_root, models_dir)
            root.mkdir(parents=True, exist_ok=True)
            target = root / spec.file_name
            tmp_target = target.with_suffix(target.suffix + ".part")
            _download_file(
                url=spec.url,
                target=tmp_target,
                proxy=proxy,
                on_progress=lambda payload: on_progress({"name": spec.name, "path": str(target), **payload}),
            )
            if tmp_target.stat().st_size <= 0:
                raise RuntimeError("Downloaded model is empty")
            tmp_target.replace(target)
            on_progress(
                {
                    "name": spec.name,
                    "message": "Downloaded",
                    "downloadedBytes": _file_size(target),
                    "speedBps": 0,
                    "path": str(target),
                }
            )
            on_done(None)
        except Exception as exc:
            on_done(f"{type(exc).__name__}: {exc}")

    thread = threading.Thread(target=_run, name="diarization-model-download", daemon=True)
    thread.start()


def delete_diarization_model(*, project_root: Path, path: str, models_dir: str | Path | None = None) -> None:
    root = diarization_models_dir(project_root, models_dir).resolve()
    target = Path(path).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Can only delete Speaker ID models inside the configured Speaker ID models directory") from exc
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
    request = Request(url, headers={"User-Agent": "MeetingScribe/1.0"})
    downloaded = 0
    last_bytes = 0
    last_ts = time.monotonic()
    with _temporary_proxy_env(proxy):
        with urlopen(request, timeout=30) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            with target.open("wb") as out:
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
                            "message": "Downloading...",
                            "downloadedBytes": downloaded,
                            "totalBytes": total,
                            "speedBps": speed_bps,
                        }
                    )


def _recommended_record(spec: DiarizationModelSpec, root: Path, downloads: dict[str, dict]) -> dict:
    path = root / spec.file_name
    download = downloads.get(spec.name) or {}
    cached = path.exists() and _file_size(path) > 0
    return {
        "name": spec.name,
        "label": spec.label,
        "fileName": spec.file_name,
        "backend": spec.backend,
        "provider": spec.provider,
        "url": spec.url,
        "path": str(path),
        "cached": cached,
        "compatible": cached,
        "status": "ready" if cached else "downloadable",
        "source": "recommended",
        "recommended": True,
        "downloadable": True,
        "deletable": cached,
        "bytes": _file_size(path),
        **_download_fields(download),
    }


def _local_record(path: Path) -> dict:
    return {
        "name": str(path),
        "label": path.name,
        "fileName": path.name,
        "backend": "sherpa_onnx",
        "provider": "cpu",
        "url": "",
        "path": str(path),
        "cached": True,
        "compatible": True,
        "status": "ready",
        "source": "local-file",
        "recommended": False,
        "downloadable": False,
        "deletable": True,
        "bytes": _file_size(path),
    }


def _download_fields(download: dict) -> dict:
    return {
        "downloading": download.get("state") == "downloading",
        "downloadDone": download.get("state") == "done",
        "downloadError": str(download.get("error") or ""),
        "downloadMessage": str(download.get("message") or ""),
        "downloadedBytes": int(download.get("downloadedBytes") or 0),
        "totalBytes": int(download.get("totalBytes") or 0),
        "speedBps": float(download.get("speedBps") or 0.0),
        "downloadUsesProxy": bool(download.get("proxy", False)),
    }


def _model_spec(name: str) -> DiarizationModelSpec:
    text = str(name or "").strip()
    for spec in RECOMMENDED_DIARIZATION_MODELS:
        if text in {spec.name, spec.file_name, spec.url}:
            return spec
    raise ValueError(f"Unknown Speaker ID model: {name}")


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size) if path.exists() else 0
    except OSError:
        return 0


@contextmanager
def _temporary_proxy_env(proxy: str) -> Iterator[None]:
    proxy_text = str(proxy or "").strip()
    if not proxy_text:
        yield
        return
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
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
