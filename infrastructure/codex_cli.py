from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from application.codex_assistant import (
    CodexAssistantPort,
    CodexAssistantRequest,
    CodexAssistantResult,
    CodexExecutionSettings,
)
from application.codex_config import CodexProfile
from application.codex_prompting import normalize_model_name, normalize_reasoning_effort
from application.local_paths import project_runtime_dir


class CodexCliRunner(CodexAssistantPort):
    def run(self, request: CodexAssistantRequest) -> CodexAssistantResult:
        t0 = time.time()
        out_path: Optional[Path] = None
        profile = request.profile
        settings = request.settings
        try:
            env = os.environ.copy()
            tmp_dir = project_runtime_dir(request.project_root, "codex")
            for key in ("TMP", "TEMP", "TMPDIR"):
                env[key] = str(tmp_dir)
            codex_home = Path(request.project_root).resolve() / ".local" / "codex_home"
            codex_home.mkdir(parents=True, exist_ok=True)
            env["CODEX_HOME"] = str(codex_home)

            proxy = str(settings.proxy or "").strip()
            if proxy:
                env["HTTP_PROXY"] = proxy
                env["HTTPS_PROXY"] = proxy
                env["ALL_PROXY"] = proxy
                env["http_proxy"] = proxy
                env["https_proxy"] = proxy
                env["all_proxy"] = proxy

            base_cmd, src = self._resolve_base_command(settings)
            if base_cmd is None:
                return self._result(
                    False,
                    profile,
                    request.original_cmd,
                    (
                        "codex executable not found. "
                        "Set codex.command in config.json (e.g. "
                        "'C:/Users/<you>/AppData/Roaming/npm/codex.cmd' or full codex.exe path)."
                    ),
                    t0,
                )

            cmd: List[str] = list(base_cmd) + ["exec", "--color", "never", "--skip-git-repo-check"]
            model_name = normalize_model_name(profile.model)
            if model_name:
                cmd.extend(["-m", model_name])
            effort = normalize_reasoning_effort(profile.reasoning_effort)
            if effort:
                cmd.extend(["-c", f'model_reasoning_effort="{effort}"'])
            if profile.codex_profile:
                cmd.extend(["-p", str(profile.codex_profile)])
            if profile.extra_args:
                cmd.extend([str(x) for x in profile.extra_args if str(x).strip()])

            out_path = tmp_dir / f"codex_last_{os.getpid()}_{uuid.uuid4().hex}.txt"
            cmd.extend(["-o", str(out_path), "-"])

            prompt_safe = request.prompt.encode("utf-8", errors="replace").decode("utf-8")
            process = subprocess.run(
                cmd,
                input=prompt_safe,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                cwd=str(request.project_root),
                env=env,
                timeout=max(10, int(settings.timeout_s)),
            )

            out_text = ""
            if out_path.exists():
                try:
                    out_text = out_path.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    out_text = ""
            if not out_text:
                out_text = (process.stdout or "").strip()

            if process.returncode != 0:
                err = (process.stderr or "").strip() or (process.stdout or "").strip()
                if not err:
                    err = f"codex exec failed with code {process.returncode}"
                else:
                    err = f"{err}\n(source={src})"
                return self._result(False, profile, request.original_cmd, err, t0)

            return self._result(True, profile, request.original_cmd, out_text or "(empty response)", t0)

        except subprocess.TimeoutExpired:
            return self._result(
                False,
                profile,
                request.original_cmd,
                f"timeout after {int(settings.timeout_s)}s",
                t0,
            )
        except FileNotFoundError:
            return self._result(
                False,
                profile,
                request.original_cmd,
                (
                    "codex executable not found at runtime. "
                    "Try setting codex.command in config.json to an explicit path."
                ),
                t0,
            )
        except Exception as e:
            return self._result(False, profile, request.original_cmd, f"{type(e).__name__}: {e}", t0)
        finally:
            if out_path is not None:
                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _resolve_base_command(self, settings: CodexExecutionSettings) -> Tuple[Optional[List[str]], str]:
        tokens = [str(x).strip() for x in settings.command_tokens if str(x).strip()]
        if not tokens:
            tokens = ["codex"]
        base = tokens[0]
        tail = tokens[1:]

        pbase = Path(base)
        if pbase.exists():
            return (self._wrap_cmd_for_windows(str(pbase), tail), "direct_path")

        found = shutil.which(base)
        if found:
            return (self._wrap_cmd_for_windows(found, tail), "path")

        names: List[str] = [base]
        if base.lower() == "codex":
            names = ["codex", "codex.exe", "codex.cmd", "codex.bat", "codex.CMD"]

        for name in names:
            found_name = shutil.which(name)
            if found_name:
                return (self._wrap_cmd_for_windows(found_name, tail), f"path:{name}")

        for directory in self._common_search_dirs(settings):
            for name in names:
                candidate = directory / name
                if candidate.exists():
                    return (self._wrap_cmd_for_windows(str(candidate), tail), f"hint:{directory}")

        try:
            probe = subprocess.run(
                ["where", "codex"],
                capture_output=True,
                text=True,
                timeout=4,
            )
            if probe.returncode == 0:
                for line in (probe.stdout or "").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    candidate = Path(line)
                    if candidate.exists():
                        return (self._wrap_cmd_for_windows(str(candidate), tail), "where")
        except Exception:
            pass

        return (None, "")

    def _common_search_dirs(self, settings: CodexExecutionSettings) -> List[Path]:
        out: List[Path] = []

        for raw in settings.path_hints:
            path = Path(str(raw).strip())
            if path and str(path).strip():
                out.append(path)

        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            out.append(Path(appdata) / "npm")

        home = Path.home()
        out.append(home / "AppData" / "Roaming" / "npm")

        ext_root = home / ".vscode" / "extensions"
        if ext_root.exists():
            try:
                exes = list(ext_root.glob("openai.chatgpt-*/bin/windows-x86_64"))
                exes.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                out.extend(exes[:5])
            except Exception:
                pass

        uniq: List[Path] = []
        seen = set()
        for path in out:
            try:
                key = str(path.resolve())
            except Exception:
                key = str(path)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(path)
        return uniq

    @staticmethod
    def _wrap_cmd_for_windows(exe_path: str, tail: List[str]) -> List[str]:
        path = str(exe_path).strip()
        suffix = Path(path).suffix.lower()
        if suffix in (".cmd", ".bat"):
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            return [comspec, "/d", "/c", path, *tail]
        return [path, *tail]

    @staticmethod
    def _result(
        ok: bool,
        profile: CodexProfile,
        original_cmd: str,
        text: str,
        t0: float,
    ) -> CodexAssistantResult:
        return CodexAssistantResult(
            ok=bool(ok),
            profile=str(profile.label),
            cmd=str(original_cmd),
            text=str(text),
            dt_s=time.time() - t0,
        )
