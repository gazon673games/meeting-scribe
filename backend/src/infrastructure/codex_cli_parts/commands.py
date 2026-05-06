from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List

from application.codex_config import CodexProfile
from application.codex_prompting import normalize_model_name, normalize_reasoning_effort


def build_exec_cmd(base_cmd: List[str], profile: CodexProfile, out_path: Path) -> List[str]:
    cmd = list(base_cmd) + ["exec", "--color", "never", "--skip-git-repo-check"]
    if model := normalize_model_name(profile.model):
        cmd.extend(["-m", model])
    if effort := normalize_reasoning_effort(profile.reasoning_effort):
        cmd.extend(["-c", f'model_reasoning_effort="{effort}"'])
    if profile.codex_profile:
        cmd.extend(["-p", str(profile.codex_profile)])
    if profile.extra_args:
        cmd.extend([str(item) for item in profile.extra_args if str(item).strip()])
    return cmd + ["-o", str(out_path), "-"]


def read_output_file(out_path: Path, proc: subprocess.CompletedProcess) -> str:
    if out_path.exists():
        try:
            if text := out_path.read_text(encoding="utf-8", errors="ignore").strip():
                return text
        except Exception:
            pass
    return (proc.stdout or "").strip()


def process_output_text(proc: subprocess.CompletedProcess) -> str:
    return ((proc.stdout or "").strip() or (proc.stderr or "").strip()).strip()


def interactive_login_cmd(base_cmd: List[str], login_args: List[str]) -> List[str]:
    if os.name != "nt":
        return list(base_cmd) + list(login_args)

    if base_cmd and Path(base_cmd[0]).name.lower() in ("cmd", "cmd.exe"):
        cmd = list(base_cmd)
        for index, token in enumerate(cmd):
            if str(token).lower() == "/c":
                cmd[index] = "/k"
                return cmd + list(login_args)
        return cmd + list(login_args)

    return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/k", *base_cmd, *login_args]


def new_console_creationflags() -> int:
    if os.name == "nt":
        return int(getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    return 0
