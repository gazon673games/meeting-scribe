from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from application.codex_assistant import (
    CodexAssistantRequest,
    CodexAssistantResult,
    CodexExecutionSettings,
)
from application.codex_config import CodexProfile
from application.codex_prompting import normalize_model_name, normalize_reasoning_effort
from application.local_paths import project_runtime_dir
from assistant.application.provider import (
    ASSISTANT_PROVIDER_CODEX,
    AssistantProviderError,
    AssistantProviderInfo,
    AssistantProviderPort,
    result_from_error,
)

_ERR_NOT_FOUND = (
    "codex executable not found. "
    "Set codex.command in config.json "
    "(e.g. 'C:/Users/<you>/AppData/Roaming/npm/codex.cmd' or full codex.exe path)."
)
_ERR_NOT_FOUND_RUNTIME = (
    "codex executable not found at runtime. "
    "Try setting codex.command in config.json to an explicit path."
)


class CodexCommandResolver:
    """Resolves a codex executable path using multiple fallback strategies."""

    def resolve(self, settings: CodexExecutionSettings) -> Optional[Tuple[List[str], str]]:
        tokens = [str(x).strip() for x in settings.command_tokens if str(x).strip()]
        base = tokens[0] if tokens else "codex"
        tail = tokens[1:] if len(tokens) > 1 else []

        if Path(base).exists():
            return self._wrap(base, tail), "direct_path"

        names = (
            ["codex", "codex.exe", "codex.cmd", "codex.bat", "codex.CMD"]
            if base.lower() == "codex" else [base]
        )
        for name in names:
            if found := shutil.which(name):
                return self._wrap(found, tail), f"path:{name}"

        for d in self._search_dirs(settings):
            for name in names:
                c = d / name
                if c.exists():
                    return self._wrap(str(c), tail), f"hint:{d}"

        return self._try_where(tail)

    def _search_dirs(self, settings: CodexExecutionSettings) -> List[Path]:
        dirs: List[Path] = [Path(str(r).strip()) for r in settings.path_hints if str(r).strip()]
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
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
                dirs.extend(bins[:5])
            except Exception:
                pass
        seen: set = set()
        unique: List[Path] = []
        for p in dirs:
            key = str(p.resolve()) if p.exists() else str(p)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    def _try_where(self, tail: List[str]) -> Optional[Tuple[List[str], str]]:
        try:
            probe = subprocess.run(["where", "codex"], capture_output=True, text=True, timeout=4)
            if probe.returncode == 0:
                for line in (probe.stdout or "").splitlines():
                    c = Path(line.strip())
                    if c.exists():
                        return self._wrap(str(c), tail), "where"
        except Exception:
            pass
        return None

    @staticmethod
    def _wrap(exe: str, tail: List[str]) -> List[str]:
        if Path(exe).suffix.lower() in (".cmd", ".bat"):
            return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", exe, *tail]
        return [exe, *tail]


class CodexCliRunner(AssistantProviderPort):
    provider_id = ASSISTANT_PROVIDER_CODEX
    provider_label = "Codex CLI"
    _resolver = CodexCommandResolver()

    def status(self, settings: CodexExecutionSettings) -> AssistantProviderInfo:
        resolved = self._resolver.resolve(settings)
        if resolved is None:
            error = _codex_not_found_error()
            return AssistantProviderInfo(
                id=self.provider_id,
                label=self.provider_label,
                available=False,
                message=error.message,
                error_code=error.code,
                retryable=error.retryable,
                suggestion=error.suggestion,
            )
        _cmd, source = resolved
        return AssistantProviderInfo(
            id=self.provider_id,
            label=self.provider_label,
            available=True,
            message=f"Found via {source}",
        )

    def run(self, request: CodexAssistantRequest) -> CodexAssistantResult:
        t0 = time.time()
        out_path: Optional[Path] = None
        profile, settings = request.profile, request.settings
        try:
            tmp_dir = project_runtime_dir(request.project_root, "codex")
            env = self._build_env(settings, request.project_root, tmp_dir)

            resolved = self._resolver.resolve(settings)
            if resolved is None:
                return self._error_result(profile, request.original_cmd, _codex_not_found_error(), t0)
            base_cmd, src = resolved

            out_path = tmp_dir / f"codex_last_{os.getpid()}_{uuid.uuid4().hex}.txt"
            cmd = self._build_cmd(base_cmd, profile, out_path)
            prompt = request.prompt.encode("utf-8", errors="replace").decode("utf-8")

            proc = subprocess.run(
                cmd, input=prompt, text=True, encoding="utf-8", errors="replace",
                capture_output=True, cwd=str(request.project_root), env=env,
                timeout=max(10, int(settings.timeout_s)),
            )
            out_text = self._read_output(out_path, proc)

            if proc.returncode != 0:
                err = (proc.stderr or "").strip() or (proc.stdout or "").strip()
                err = f"{err}\n(source={src})" if err else f"codex exec failed with code {proc.returncode}"
                return self._error_result(profile, request.original_cmd, _classify_codex_error(err), t0)
            return self._result(True, profile, request.original_cmd, out_text or "(empty response)", t0)

        except subprocess.TimeoutExpired:
            return self._error_result(
                profile,
                request.original_cmd,
                AssistantProviderError(
                    code="timeout",
                    message=f"timeout after {int(settings.timeout_s)}s",
                    retryable=True,
                    suggestion="Increase assistant timeout or use a faster model/profile.",
                ),
                t0,
            )
        except FileNotFoundError:
            return self._error_result(profile, request.original_cmd, _codex_not_found_runtime_error(), t0)
        except Exception as e:
            return self._error_result(
                profile,
                request.original_cmd,
                AssistantProviderError(
                    code="unknown_error",
                    message=f"{type(e).__name__}: {e}",
                    retryable=True,
                ),
                t0,
            )
        finally:
            if out_path is not None:
                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass

    @staticmethod
    def _build_env(settings: CodexExecutionSettings, project_root, tmp_dir: Path) -> dict:
        env = os.environ.copy()
        for key in ("TMP", "TEMP", "TMPDIR"):
            env[key] = str(tmp_dir)
        codex_home = Path(project_root).resolve() / ".local" / "codex_home"
        codex_home.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(codex_home)
        if proxy := str(settings.proxy or "").strip():
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                env[k] = proxy
        return env

    @staticmethod
    def _build_cmd(base_cmd: List[str], profile: CodexProfile, out_path: Path) -> List[str]:
        cmd = list(base_cmd) + ["exec", "--color", "never", "--skip-git-repo-check"]
        if model := normalize_model_name(profile.model):
            cmd.extend(["-m", model])
        if effort := normalize_reasoning_effort(profile.reasoning_effort):
            cmd.extend(["-c", f'model_reasoning_effort="{effort}"'])
        if profile.codex_profile:
            cmd.extend(["-p", str(profile.codex_profile)])
        if profile.extra_args:
            cmd.extend([str(x) for x in profile.extra_args if str(x).strip()])
        return cmd + ["-o", str(out_path), "-"]

    @staticmethod
    def _read_output(out_path: Path, proc: subprocess.CompletedProcess) -> str:
        if out_path.exists():
            try:
                if text := out_path.read_text(encoding="utf-8", errors="ignore").strip():
                    return text
            except Exception:
                pass
        return (proc.stdout or "").strip()

    @staticmethod
    def _result(ok: bool, profile: CodexProfile, original_cmd: str, text: str, t0: float) -> CodexAssistantResult:
        return CodexAssistantResult(
            ok=bool(ok),
            profile=str(profile.label),
            cmd=str(original_cmd),
            text=str(text),
            dt_s=time.time() - t0,
            provider=ASSISTANT_PROVIDER_CODEX,
            model=str(profile.model or ""),
        )

    @staticmethod
    def _error_result(
        profile: CodexProfile,
        original_cmd: str,
        error: AssistantProviderError,
        t0: float,
    ) -> CodexAssistantResult:
        return result_from_error(
            profile=profile,
            cmd=original_cmd,
            provider=ASSISTANT_PROVIDER_CODEX,
            model=str(profile.model or ""),
            error=error,
            started_at=t0,
        )


def _codex_not_found_error() -> AssistantProviderError:
    return AssistantProviderError(
        code="codex_not_found",
        message=_ERR_NOT_FOUND,
        retryable=False,
        suggestion="Install Codex CLI or set codex.command to the full executable path.",
    )


def _codex_not_found_runtime_error() -> AssistantProviderError:
    return AssistantProviderError(
        code="codex_not_found",
        message=_ERR_NOT_FOUND_RUNTIME,
        retryable=False,
        suggestion="Set codex.command to an explicit codex executable path.",
    )


def _classify_codex_error(message: str) -> AssistantProviderError:
    text = str(message or "").strip()
    lower = text.lower()
    if any(marker in lower for marker in ("rate limit", "rate_limit", "too many requests", " 429", "(429", "status 429")):
        return AssistantProviderError(
            code="rate_limited",
            message=text,
            retryable=True,
            suggestion="Wait a little or switch to another assistant profile/model.",
        )
    if any(marker in lower for marker in ("login", "auth", "unauthorized", "forbidden", "api key", "401", "403")):
        return AssistantProviderError(
            code="auth_error",
            message=text,
            retryable=False,
            suggestion="Check Codex login/API credentials and try again.",
        )
    if any(
        marker in lower
        for marker in (
            "enotfound",
            "eai_again",
            "dns",
            "network",
            "connection",
            "connect",
            "proxy",
            "timed out",
            "timeout",
            "tls",
            "ssl",
        )
    ):
        return AssistantProviderError(
            code="network_error",
            message=text,
            retryable=True,
            suggestion="Check internet access or proxy settings.",
        )
    if "model" in lower and any(marker in lower for marker in ("not found", "unavailable", "unsupported", "unknown")):
        return AssistantProviderError(
            code="model_unavailable",
            message=text,
            retryable=False,
            suggestion="Choose another Codex model/profile.",
        )
    return AssistantProviderError(
        code="provider_crash",
        message=text,
        retryable=True,
        suggestion="Retry the request; if it repeats, check the assistant console output.",
    )
