#!/usr/bin/env python3
"""
Standalone batch transcription + optional speaker diarization.

CLI:
    python tools/batch_scribe.py meeting.mp4 --profile Quality --language ru
    python tools/batch_scribe.py meeting.mp4 --diar --word-by-word --out-dir ./out

Library:
    from pathlib import Path
    from batch_scribe import scribe_to_srt

    result = scribe_to_srt(
        Path("meeting.mp4"),
        Path("meeting.srt"),
        profile_name="Quality",
        model="large-v3",
        language="ru",
        word_by_word=True,
    )
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


def _bootstrap(project_root: Optional[str] = None, models_dir: Optional[str] = None) -> None:
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    src = (root / "backend" / "src").resolve()
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        from application.local_paths import configure_project_local_io

        configure_project_local_io(root, models_dir=models_dir)
    except Exception:
        pass


_bootstrap()


from tools.batch_scribe_parts.cli import _parse_asr_options, build_parser, main as _main  # noqa: E402
from tools.batch_scribe_parts.constants import OUTPUT_FORMATS, PROFILES  # noqa: E402
from tools.batch_scribe_parts.entrypoint import BatchScriber, scribe_to_srt  # noqa: E402
from tools.batch_scribe_parts.output import append_srt_segment, write_output  # noqa: E402
from tools.batch_scribe_parts.profiles import build_profile, profile_from_request  # noqa: E402
from tools.batch_scribe_parts.request import BatchScribeRequest, BatchScribeResult  # noqa: E402
from tools.batch_scribe_parts.scribe import Scribe  # noqa: E402


__all__ = [
    "BatchScribeRequest",
    "BatchScribeResult",
    "BatchScriber",
    "OUTPUT_FORMATS",
    "PROFILES",
    "Scribe",
    "_parse_asr_options",
    "append_srt_segment",
    "build_parser",
    "build_profile",
    "profile_from_request",
    "scribe_to_srt",
    "write_output",
]


def main(argv=None) -> None:
    _main(argv, bootstrap=_bootstrap)


if __name__ == "__main__":
    main()
