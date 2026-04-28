from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def project_local_root(project_root: Path) -> Path:
    return Path(project_root).resolve() / ".local"


def project_logs_dir(project_root: Path, *, create: bool = False, read_legacy: bool = False) -> Path:
    return _project_named_dir(project_root, "logs", create=create, read_legacy=read_legacy)


def project_human_logs_dir(project_root: Path, *, create: bool = False, read_legacy: bool = False) -> Path:
    return _project_named_dir(project_root, "human_logs", create=create, read_legacy=read_legacy)


def project_recordings_dir(project_root: Path, *, create: bool = False) -> Path:
    return _project_named_dir(project_root, "recordings", create=create, read_legacy=False)


def project_runtime_dir(project_root: Path, name: str) -> Path:
    safe_name = Path(str(name or "tmp")).name or "tmp"
    path = project_local_root(project_root) / "tmp" / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_project_local_io(project_root: Path, *, models_dir: Path | str | None = None) -> None:
    root = Path(project_root).resolve()
    models_dir = Path(models_dir).expanduser().resolve() if models_dir else root / "models"
    tmp_dir = project_local_root(root) / "tmp"

    models_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cache_env = {
        "HF_HUB_CACHE": models_dir,
        "HF_HOME": models_dir / "hf_home",
        "TRANSFORMERS_CACHE": models_dir / "transformers",
        "TORCH_HOME": models_dir / "torch",
        "NEMO_CACHE_DIR": models_dir / "nemo",
        "XDG_CACHE_HOME": models_dir / "xdg",
    }
    for key, path in cache_env.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)

    for key in ("TMP", "TEMP", "TMPDIR"):
        os.environ[key] = str(tmp_dir)
    tempfile.tempdir = str(tmp_dir)


def _project_named_dir(project_root: Path, name: str, *, create: bool, read_legacy: bool) -> Path:
    safe_name = Path(str(name or "tmp")).name or "tmp"
    preferred = project_local_root(project_root) / safe_name
    if create:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    if preferred.exists():
        return preferred
    legacy = Path(project_root).resolve() / safe_name
    if read_legacy and legacy.exists():
        return legacy
    return preferred
