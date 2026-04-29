from __future__ import annotations

import os
import re
import shutil
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen


PREFERRED_GGUF_QUANTS = (
    "Q4_K_M",
    "Q4_K_L",
    "Q5_K_M",
    "Q4_K_S",
    "Q8_0",
    "IQ4_XS",
)


def llm_models_dir(project_root: Path, models_dir: str | Path | None = None) -> Path:
    if models_dir:
        return Path(models_dir).expanduser().resolve()
    return Path(project_root).resolve() / "models" / "llm"


def list_llm_models(
    *,
    project_root: Path,
    models_dir: str | Path | None = None,
    downloads: Optional[dict[str, dict]] = None,
) -> dict:
    root = llm_models_dir(project_root, models_dir)
    root.mkdir(parents=True, exist_ok=True)
    active = downloads or {}
    models = [_local_record(path, active) for path in sorted(root.rglob("*.gguf"), key=lambda item: item.name.lower())]

    known = set()
    for model in models:
        known.add(str(model.get("name") or ""))
        known.add(str(model.get("fileName") or ""))
        model_path = str(model.get("path") or "")
        if model_path:
            known.add(Path(model_path).parent.name)
    for name, record in active.items():
        if str(name) in known:
            continue
        models.append(_download_record(str(name), record))
    return {"models": models, "modelsDir": str(root)}


def download_llm_model_async(
    *,
    name: str,
    project_root: Path,
    on_progress: Callable[[dict], None],
    on_done: Callable[[Optional[str]], None],
    models_dir: str | Path | None = None,
    proxy: str = "",
) -> None:
    source = parse_llm_source(name)

    def _run() -> None:
        try:
            resolved_source = _resolve_repo_source(source, proxy=proxy)
            root = llm_models_dir(project_root, models_dir)
            root.mkdir(parents=True, exist_ok=True)
            target_dir = root / resolved_source.folder
            target = target_dir / resolved_source.filename
            target_dir.mkdir(parents=True, exist_ok=True)

            on_progress({"message": f"Downloading {resolved_source.filename}...", "path": str(target), "downloadedBytes": _dir_size(target_dir), "speedBps": 0})
            if resolved_source.repo_id:
                _download_hf_file(resolved_source, target, proxy=proxy, on_progress=on_progress)
            else:
                _download_url(resolved_source.url, target, proxy=proxy, on_progress=on_progress)

            if not target.exists() or target.stat().st_size <= 0:
                raise RuntimeError("Downloaded GGUF file is empty")
            on_progress({"message": "Downloaded", "path": str(target), "downloadedBytes": _file_size(target), "speedBps": 0})
            on_done(None)
        except Exception as exc:
            on_done(f"{type(exc).__name__}: {exc}")

    threading.Thread(target=_run, name="llm-model-download", daemon=True).start()


def delete_llm_model(*, project_root: Path, path: str, models_dir: str | Path | None = None) -> None:
    root = llm_models_dir(project_root, models_dir).resolve()
    target = Path(path).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("Can only delete language models inside the configured language models directory") from exc
    if target.exists():
        target.unlink()
    parent = target.parent
    if parent != root and parent.exists() and not any(parent.iterdir()):
        parent.rmdir()


class LlmSource:
    def __init__(self, *, filename: str, folder: str, repo_id: str = "", url: str = "") -> None:
        self.filename = filename
        self.folder = folder
        self.repo_id = repo_id
        self.url = url


def parse_llm_source(raw: str) -> LlmSource:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Language model source is empty")
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc.lower().endswith("huggingface.co"):
        return _huggingface_url_source(parsed)
    if parsed.scheme in {"http", "https"}:
        filename = _safe_filename(Path(unquote(parsed.path)).name)
        if not filename.lower().endswith(".gguf"):
            raise ValueError("Direct language model URL must point to a .gguf file")
        return LlmSource(filename=filename, folder=filename.removesuffix(".gguf"), url=text)

    repo_id, filename = _repo_and_filename(text)
    return LlmSource(filename=_safe_filename(filename) if filename else "", folder=_safe_folder(repo_id), repo_id=repo_id)


def _huggingface_url_source(parsed) -> LlmSource:  # noqa: ANN001
    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError("Hugging Face URL must include a repo")
    repo_id = f"{parts[0]}/{parts[1]}"
    if len(parts) == 2 or (len(parts) >= 3 and parts[2] == "tree"):
        return LlmSource(filename="", folder=_safe_folder(repo_id), repo_id=repo_id)
    if len(parts) < 5 or parts[2] not in {"resolve", "blob"}:
        raise ValueError("Hugging Face URL must include a repo or .gguf file path")
    filename = "/".join(parts[4:])
    if not filename.lower().endswith(".gguf"):
        raise ValueError("Hugging Face language model file must be .gguf")
    return LlmSource(filename=filename, folder=_safe_folder(repo_id), repo_id=repo_id)


def _repo_and_filename(text: str) -> tuple[str, str]:
    compact = re.sub(r"\s+", "/", text.strip())
    parts = [part for part in compact.replace("\\", "/").split("/") if part]
    if len(parts) < 2:
        raise ValueError("Use repo or repo/file.gguf, for example org/model or org/model/model-Q4_K_M.gguf")
    repo_id = f"{parts[0]}/{parts[1]}"
    if len(parts) == 2:
        return repo_id, ""
    filename = "/".join(parts[2:])
    if not filename.lower().endswith(".gguf"):
        raise ValueError("Language model file must be .gguf")
    return repo_id, filename


def _resolve_repo_source(source: LlmSource, *, proxy: str) -> LlmSource:
    if not source.repo_id or source.filename:
        return source
    from huggingface_hub import model_info  # type: ignore

    with _temporary_proxy_env(proxy):
        info = model_info(source.repo_id)
    files = [
        str(getattr(item, "rfilename", "") or (item.get("rfilename") if isinstance(item, dict) else "")).strip()
        for item in list(getattr(info, "siblings", []) or [])
    ]
    filename = _choose_gguf_file(files)
    return LlmSource(filename=filename, folder=source.folder, repo_id=source.repo_id)


def _choose_gguf_file(files: list[str]) -> str:
    ggufs = [name for name in files if name.lower().endswith(".gguf")]
    if not ggufs:
        raise ValueError("Hugging Face repo does not contain .gguf files")
    for quant in PREFERRED_GGUF_QUANTS:
        for name in ggufs:
            if quant.lower() in Path(name).name.lower():
                return name
    return sorted(ggufs, key=lambda value: value.lower())[0]


def _download_hf_file(source: LlmSource, target: Path, *, proxy: str, on_progress: Callable[[dict], None]) -> None:
    from huggingface_hub import hf_hub_url  # type: ignore

    url = hf_hub_url(repo_id=source.repo_id, filename=source.filename)
    _download_url(url, target, proxy=proxy, on_progress=on_progress, headers=_hf_headers())


def _hf_headers() -> dict[str, str]:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def _download_url(
    url: str,
    target: Path,
    *,
    proxy: str,
    on_progress: Callable[[dict], None],
    headers: Optional[dict[str, str]] = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "MeetingScribe/1.0", **(headers or {})})
    downloaded = 0
    last_bytes = 0
    last_ts = time.monotonic()
    tmp_target = target.with_suffix(target.suffix + ".part")
    with _temporary_proxy_env(proxy):
        with urlopen(request, timeout=30) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            with tmp_target.open("wb") as out:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    speed = max(0.0, (downloaded - last_bytes) / max(1e-6, now - last_ts))
                    last_bytes = downloaded
                    last_ts = now
                    on_progress({"message": "Downloading...", "path": str(target), "downloadedBytes": downloaded, "totalBytes": total, "speedBps": speed})
    tmp_target.replace(target)


def _local_record(path: Path, downloads: dict[str, dict]) -> dict:
    name = path.stem
    download = downloads.get(name) or downloads.get(path.name) or {}
    return {
        "name": name,
        "label": path.name,
        "fileName": path.name,
        "path": str(path),
        "modelAlias": name,
        "cached": True,
        "compatible": True,
        "status": "ready",
        "source": "local-file",
        "downloadable": False,
        "deletable": True,
        "bytes": _file_size(path),
        **_download_fields(download),
    }


def _download_record(name: str, download: dict) -> dict:
    return {
        "name": name,
        "label": name,
        "fileName": name,
        "path": str(download.get("path") or ""),
        "modelAlias": Path(name).stem,
        "cached": False,
        "compatible": False,
        "status": "downloading" if download.get("state") == "downloading" else "error",
        "source": "download",
        "downloadable": False,
        "deletable": False,
        "bytes": 0,
        **_download_fields(download),
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


def _safe_filename(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if "/" in text:
        return "/".join(_safe_filename(part) for part in text.split("/") if part)
    if not text or text in {".", ".."}:
        raise ValueError("Invalid language model filename")
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", text)


def _safe_folder(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-") or "model"


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size) if path.exists() else 0
    except OSError:
        return 0


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(_file_size(item) for item in path.rglob("*") if item.is_file())


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
