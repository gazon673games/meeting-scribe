from __future__ import annotations

import sys
from pathlib import Path

from application.local_paths import application_root, configure_project_local_io


# Keep model caches and temp files beside the app, not in user/global temp dirs.
# Must run before huggingface_hub, torch, NeMo, faster_whisper, or Codex imports.
_runtime_root = application_root()
configure_project_local_io(_runtime_root)

from app_bootstrap import main
from ui.app import ensure_runtime_config


if __name__ == "__main__":
    if "--repair-config" in sys.argv:
        root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
        ensure_runtime_config(root, root / "config.json")
        raise SystemExit(0)
    if "--smoke-import" in sys.argv:
        raise SystemExit(0)
    main()
