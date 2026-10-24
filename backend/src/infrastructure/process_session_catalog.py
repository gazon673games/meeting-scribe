from __future__ import annotations

import sys
from typing import Any

from infrastructure.process_session_catalog_parts import (
    list_linux_sessions,
    list_windows_session_groups,
)


def is_per_process_audio_supported() -> bool:
    if sys.platform == 'win32':
        try:
            version = sys.getwindowsversion()
            return version.major > 10 or (
                version.major == 10 and version.build >= 20348
            )
        except (AttributeError, OSError):
            return False
    if sys.platform == 'linux':
        import shutil

        return bool(shutil.which('pactl') and shutil.which('parec'))
    return False


def list_process_sessions() -> list[dict[str, Any]]:
    return [
        session
        for group in list_process_session_groups()
        for session in group.get('sessions', [])
    ]


def list_process_session_groups() -> list[dict[str, Any]]:
    if sys.platform == 'win32':
        return list_windows_session_groups()
    if sys.platform == 'linux':
        sessions = list_linux_sessions()
        return (
            [
                {
                    'id': 'linux-default',
                    'label': 'Default output',
                    'sessions': sessions,
                }
            ]
            if sessions
            else []
        )
    return []
