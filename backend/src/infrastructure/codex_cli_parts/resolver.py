from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from application.codex_assistant import CodexExecutionSettings


class CodexCommandResolver:
    """Resolves a codex executable path using multiple fallback strategies."""

    def resolve(self, settings: CodexExecutionSettings) -> Optional[Tuple[List[str], str]]:
        tokens = [str(item).strip() for item in settings.command_tokens if str(item).strip()]
        base = tokens[0] if tokens else "codex"
        tail = tokens[1:] if len(tokens) > 1 else []

        if Path(base).exists():
            return self._wrap(base, tail), "direct_path"

        names = (
            ["codex", "codex.exe", "codex.cmd", "codex.bat", "codex.CMD"]
            if base.lower() == "codex"
            else [base]
        )

        for name in names:
            if found := shutil.which(name):
                return self._wrap(found, tail), f"path:{name}"

        for search_dir in self._search_dirs(settings):
            for name in names:
                candidate = search_dir / name
                if candidate.exists():
                    return self._wrap(str(candidate), tail), f"hint:{search_dir}"

        return self._try_where(tail)

    def _search_dirs(self, settings: CodexExecutionSettings) -> List[Path]:
        dirs: List[Path] = [Path(str(raw).strip()) for raw in settings.path_hints if str(raw).strip()]
        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            dirs.append(Path(appdata) / "npm")
        home = Path.home()
        dirs.append(home / "AppData" / "Roaming" / "npm")

        ext_root = home / ".vscode" / "extensions"
        if ext_root.exists():
            try:
                bins = sorted(
                    ext_root.glob("openai.chatgpt-*/bin/windows-x86_64"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                dirs.extend(bins[:5])
            except Exception:
                pass

        seen: set[str] = set()
        unique: List[Path] = []
        for path in dirs:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _try_where(self, tail: List[str]) -> Optional[Tuple[List[str], str]]:
        try:
            probe = subprocess.run(["where", "codex"], capture_output=True, text=True, timeout=4)
            if probe.returncode == 0:
                for line in (probe.stdout or "").splitlines():
                    candidate = Path(line.strip())
                    if candidate.exists():
                        return self._wrap(str(candidate), tail), "where"
        except Exception:
            pass
        return None

    @staticmethod
    def _wrap(exe: str, tail: List[str]) -> List[str]:
        if Path(exe).suffix.lower() in (".cmd", ".bat"):
            return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", exe, *tail]
        return [exe, *tail]
