from __future__ import annotations

import os
import sys
from pathlib import Path

# Point HuggingFace cache to project-local models/ folder.
# Must be set before any huggingface_hub or faster_whisper imports.
_models_dir = Path(__file__).resolve().parent / "models"
_models_dir.mkdir(exist_ok=True)
os.environ.setdefault("HF_HUB_CACHE", str(_models_dir))

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
