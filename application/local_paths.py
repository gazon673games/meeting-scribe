from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def project_runtime_dir(project_root: Path, name: str) -> Path:
    safe_name = Path(str(name or "tmp")).name or "tmp"
    path = Path(project_root).resolve() / "tmp" / safe_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_project_local_io(project_root: Path) -> None:
    root = Path(project_root).resolve()
    models_dir = root / "models"
    tmp_dir = root / "tmp"

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
