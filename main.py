from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_ROOT = _PROJECT_ROOT / "src"
if _SRC_ROOT.exists():
    src_text = str(_SRC_ROOT)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)

from application.local_paths import configure_project_local_io
from settings.infrastructure.json_config_repository import JsonConfigRepository
from settings.infrastructure.runtime_config import ensure_runtime_config


def _runtime_root() -> Path:
    return Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else _PROJECT_ROOT


def _repair_config() -> None:
    root = _runtime_root()
    configure_project_local_io(root)
    ensure_runtime_config(root, JsonConfigRepository(root / "config.json"))


def _run_backend() -> None:
    from main_electron_backend import main as backend_main

    backend_main()


def _run_legacy_qt() -> None:
    configure_project_local_io(_runtime_root())
    from app_bootstrap import main as qt_main

    qt_main()


def _run_electron_dev() -> None:
    if not (_PROJECT_ROOT / "package.json").exists():
        raise SystemExit("Electron package.json was not found.")
    subprocess.run(["npm.cmd" if sys.platform == "win32" else "npm", "run", "dev"], cwd=_PROJECT_ROOT, check=True)


if __name__ == "__main__":
    if "--repair-config" in sys.argv:
        _repair_config()
        raise SystemExit(0)
    if "--smoke-import" in sys.argv:
        raise SystemExit(0)
    if "--backend" in sys.argv:
        _run_backend()
    elif "--qt" in sys.argv:
        _run_legacy_qt()
    else:
        _run_electron_dev()
