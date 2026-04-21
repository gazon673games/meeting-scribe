"""Test suite bootstrap for meeting-scribe."""

from __future__ import annotations

import sys
from pathlib import Path


_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
src_text = str(_SRC_ROOT)
if src_text not in sys.path:
    sys.path.insert(0, src_text)
