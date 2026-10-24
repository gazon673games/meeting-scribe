from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from application.codex_assistant import CodexExecutionSettings

_logger = logging.getLogger(__name__)
_CODEX_NAMES = ('codex', 'codex.exe', 'codex.cmd', 'codex.bat', 'codex.CMD')


class CodexCommandResolver:
    """Resolves a codex executable path using multiple fallback strategies."""

    def resolve(
        self, settings: CodexExecutionSettings
    ) -> tuple[list[str], str] | None:
        tokens = [
            str(item).strip()
            for item in settings.command_tokens
            if str(item).strip()
        ]
        base = tokens[0] if tokens else 'codex'
        tail = tokens[1:] if len(tokens) > 1 else []

        if Path(base).exists():
            return self._wrap(base, tail), 'direct_path'

        names = _CODEX_NAMES if base.lower() == 'codex' else (base,)

        for name in names:
            if found := shutil.which(name):
                return self._wrap(found, tail), f'path:{name}'

        for search_dir in self._search_dirs(settings):
            for name in names:
                candidate = search_dir / name
                if candidate.exists():
                    return self._wrap(
                        str(candidate), tail
                    ), f'hint:{search_dir}'

        return self._try_where(tail, executable=base)

    def _search_dirs(self, settings: CodexExecutionSettings) -> list[Path]:
        dirs: list[Path] = [
            Path(str(raw).strip())
            for raw in settings.path_hints
            if str(raw).strip()
        ]
        appdata = os.environ.get('APPDATA', '').strip()
        if appdata:
            dirs.append(Path(appdata) / 'npm')
        home = Path.home()
        dirs.append(home / 'AppData' / 'Roaming' / 'npm')

        ext_root = home / '.vscode' / 'extensions'
        if ext_root.exists():
            try:
                bins = sorted(
                    ext_root.glob('openai.chatgpt-*/bin/windows-x86_64'),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                dirs.extend(bins[:5])
            except OSError:
                _logger.debug(
                    'Best-effort operation failed in _search_dirs',
                    exc_info=True,
                )

        seen: set[str] = set()
        unique: list[Path] = []
        for path in dirs:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _try_where(
        self, tail: list[str], *, executable: str = 'codex'
    ) -> tuple[list[str], str] | None:
        try:
            probe = subprocess.run(
                ['where', executable], capture_output=True, text=True, timeout=4
            )
            if probe.returncode == 0:
                for line in (probe.stdout or '').splitlines():
                    candidate = Path(line.strip())
                    if candidate.exists():
                        return self._wrap(str(candidate), tail), 'where'
        except (OSError, subprocess.SubprocessError):
            _logger.debug(
                'Best-effort operation failed in _try_where', exc_info=True
            )
        return None

    @staticmethod
    def _wrap(exe: str, tail: list[str]) -> list[str]:
        if Path(exe).suffix.lower() in ('.cmd', '.bat'):
            return [
                os.environ.get('COMSPEC', 'cmd.exe'),
                '/d',
                '/c',
                exe,
                *tail,
            ]
        return [exe, *tail]
