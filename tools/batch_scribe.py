#!/usr/bin/env python3
"""
Standalone batch transcription + optional speaker diarization.

CLI:
    python tools/batch_scribe.py meeting.mp4 --profile Quality --language ru
    python tools/batch_scribe.py meeting.mp4 --diar --word-by-word --out-dir ./out

Library:
    from pathlib import Path
    from tools.batch_scribe import scribe_to_srt

    result = scribe_to_srt(
        Path('meeting.mp4'),
        Path('meeting.srt'),
        profile_name='Quality',
        model='large-v3',
        language='ru',
        word_by_word=True,
    )
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


def _configure_import_paths(project_root: str | Path | None = None) -> Path:
    project_path = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    root = project_path.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    source_root = root / 'backend' / 'src'
    if source_root.is_dir() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    return root


def _bootstrap(
    project_root: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> None:
    root = _configure_import_paths(project_root)
    from application.local_paths import configure_project_local_io

    configure_project_local_io(root, models_dir=models_dir)


# Direct script execution needs project imports before loading the CLI facade.
_configure_import_paths()


from tools.batch_scribe_parts.cli import (  # noqa: E402
    _parse_asr_options,
    build_parser,
)
from tools.batch_scribe_parts.cli import (  # noqa: E402
    main as _run_cli,
)
from tools.batch_scribe_parts.constants import (  # noqa: E402
    OUTPUT_FORMATS,
    PROFILES,
)
from tools.batch_scribe_parts.entrypoint import (  # noqa: E402
    BatchScriber,
    scribe_to_srt,
)
from tools.batch_scribe_parts.output import (  # noqa: E402
    append_srt_segment,
    write_output,
)
from tools.batch_scribe_parts.profiles import (  # noqa: E402
    build_profile,
    profile_from_request,
)
from tools.batch_scribe_parts.request import (  # noqa: E402
    BatchScribeRequest,
    BatchScribeResult,
)
from tools.batch_scribe_parts.scribe import Scribe  # noqa: E402

__all__ = (
    'BatchScribeRequest',
    'BatchScribeResult',
    'BatchScriber',
    'OUTPUT_FORMATS',
    'PROFILES',
    'Scribe',
    '_parse_asr_options',
    'append_srt_segment',
    'build_parser',
    'build_profile',
    'profile_from_request',
    'scribe_to_srt',
    'write_output',
)


def main(argv: Sequence[str] | None = None) -> None:
    _run_cli(argv, bootstrap=_bootstrap)


if __name__ == '__main__':
    main()
