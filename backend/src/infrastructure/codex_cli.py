from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
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
    AssistantProviderLoginResult,
    AssistantProviderPingResult,
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
_OPENAI_API_PING_URL = "https://api.openai.com/v1/models"


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
    _status_ttl_s = 3.0

    def __init__(self) -> None:
        self._status_cache_key: tuple | None = None
        self._status_cache_ts = 0.0
        self._status_cache_info: AssistantProviderInfo | None = None

    def status(self, settings: CodexExecutionSettings) -> AssistantProviderInfo:
        project_root = _settings_project_root(settings)
        cache_key = self._status_key(settings, project_root)
        if self._status_cache_key == cache_key and self._status_cache_info is not None:
            if time.monotonic() - self._status_cache_ts <= self._status_ttl_s:
                return self._status_cache_info

        resolved = self._resolver.resolve(settings)
        if resolved is None:
            error = _codex_not_found_error()
            return self._cache_status(cache_key, AssistantProviderInfo(
                id=self.provider_id,
                label=self.provider_label,
                available=False,
                message=error.message,
                error_code=error.code,
                retryable=error.retryable,
                suggestion=error.suggestion,
                login_supported=False,
            ))

        base_cmd, source = resolved
        local_home = str(_codex_home(project_root))
        try:
            tmp_dir = project_runtime_dir(project_root, "codex")
            proc = subprocess.run(
                list(base_cmd) + ["login", "status"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(project_root),
                env=self._build_env(settings, project_root, tmp_dir),
                timeout=6,
            )
        except subprocess.TimeoutExpired:
            error = AssistantProviderError(
                code="provider_status_timeout",
                message=f"Codex CLI was found via {source}, but login status timed out.",
                retryable=True,
                suggestion="Try again or run codex login status manually in a terminal.",
            )
            return self._cache_status(cache_key, _provider_info_from_error(error, local_home=local_home))
        except FileNotFoundError:
            error = _codex_not_found_runtime_error()
            return self._cache_status(cache_key, _provider_info_from_error(error, local_home=local_home))

        status_text = _process_output_text(proc)
        if proc.returncode == 0:
            detail = status_text or "Codex login is ready"
            return self._cache_status(cache_key, AssistantProviderInfo(
                id=self.provider_id,
                label=self.provider_label,
                available=True,
                message=f"Found via {source}; {detail}",
                login_supported=True,
                local_home=local_home,
            ))

        error = _classify_codex_error(_status_error_text(status_text, source=source, returncode=proc.returncode))
        suggestion = error.suggestion
        if error.code == "auth_error":
            suggestion = "Authorize the local Codex profile before using AI Assistant."
        return self._cache_status(cache_key, AssistantProviderInfo(
            id=self.provider_id,
            label=self.provider_label,
            available=False,
            message=error.message,
            error_code=error.code,
            retryable=error.retryable,
            suggestion=suggestion,
            auth_required=error.code == "auth_error",
            login_supported=True,
            local_home=local_home,
        ))

    def start_login(
        self,
        settings: CodexExecutionSettings,
        *,
        device_auth: bool = False,
    ) -> AssistantProviderLoginResult:
        project_root = _settings_project_root(settings)
        resolved = self._resolver.resolve(settings)
        local_home = str(_codex_home(project_root))
        self._status_cache_key = None
        self._status_cache_info = None

        if resolved is None:
            error = _codex_not_found_error()
            return AssistantProviderLoginResult(
                id=self.provider_id,
                label=self.provider_label,
                started=False,
                message=error.message,
                error_code=error.code,
                suggestion=error.suggestion,
                local_home=local_home,
            )

        base_cmd, source = resolved
        tmp_dir = project_runtime_dir(project_root, "codex")
        login_args = ["login"]
        if device_auth:
            login_args.append("--device-auth")
        cmd = _interactive_login_cmd(list(base_cmd), login_args)
        subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=self._build_env(settings, project_root, tmp_dir),
            creationflags=_new_console_creationflags(),
        )
        return AssistantProviderLoginResult(
            id=self.provider_id,
            label=self.provider_label,
            started=True,
            message=f"Opened Codex login via {source}.",
            local_home=local_home,
        )

    def ping(self, settings: CodexExecutionSettings) -> AssistantProviderPingResult:
        request = urllib.request.Request(
            _OPENAI_API_PING_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": "meeting-scribe-codex-ping/1.0",
            },
            method="GET",
        )
        opener = _url_opener(settings)
        timeout_s = min(8, max(3, int(getattr(settings, "timeout_s", 5) or 5)))
        try:
            response = opener.open(request, timeout=timeout_s)
            try:
                status_code = int(getattr(response, "status", getattr(response, "code", 0)) or 0)
            finally:
                close_fn = getattr(response, "close", None)
                if callable(close_fn):
                    close_fn()
            return AssistantProviderPingResult(
                id=self.provider_id,
                label=self.provider_label,
                ok=True,
                message=f"OpenAI API is reachable (HTTP {status_code or 200}).",
                status_code=status_code or 200,
            )
        except urllib.error.HTTPError as exc:
            status_code = int(getattr(exc, "code", 0) or 0)
            if status_code < 500:
                return AssistantProviderPingResult(
                    id=self.provider_id,
                    label=self.provider_label,
                    ok=True,
                    message=f"OpenAI API is reachable (HTTP {status_code}); auth is checked separately.",
                    status_code=status_code,
                )
            return AssistantProviderPingResult(
                id=self.provider_id,
                label=self.provider_label,
                ok=False,
                message=f"OpenAI API responded with HTTP {status_code}.",
                error_code="api_unavailable",
                retryable=True,
                suggestion="Try again later or check OpenAI service availability.",
                status_code=status_code,
            )
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            return AssistantProviderPingResult(
                id=self.provider_id,
                label=self.provider_label,
                ok=False,
                message=f"{type(exc).__name__}: {exc}",
                error_code="network_error",
                retryable=True,
                suggestion="Check internet access, DNS, firewall, or proxy settings.",
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
        codex_home = _codex_home(Path(project_root))
        codex_home.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(codex_home)
        if proxy := str(settings.proxy or "").strip():
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                env[k] = proxy
        return env

    def _status_key(self, settings: CodexExecutionSettings, project_root: Path) -> tuple:
        return (
            tuple(str(x) for x in settings.command_tokens),
            tuple(str(x) for x in settings.path_hints),
            str(settings.proxy or ""),
            str(project_root.resolve()),
        )

    def _cache_status(self, key: tuple, info: AssistantProviderInfo) -> AssistantProviderInfo:
        self._status_cache_key = key
        self._status_cache_ts = time.monotonic()
        self._status_cache_info = info
        return info

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


def _settings_project_root(settings: CodexExecutionSettings) -> Path:
    raw = getattr(settings, "project_root", None)
    return Path(raw).resolve() if raw else Path.cwd().resolve()


def _codex_home(project_root: Path) -> Path:
    return Path(project_root).resolve() / ".local" / "codex_home"


def _process_output_text(proc: subprocess.CompletedProcess) -> str:
    return ((proc.stdout or "").strip() or (proc.stderr or "").strip()).strip()


def _status_error_text(text: str, *, source: str, returncode: int) -> str:
    clean = str(text or "").strip()
    if clean:
        return f"{clean}\n(source={source})"
    return f"codex login status failed with code {returncode}\n(source={source})"


def _provider_info_from_error(error: AssistantProviderError, *, local_home: str = "") -> AssistantProviderInfo:
    return AssistantProviderInfo(
        id=ASSISTANT_PROVIDER_CODEX,
        label="Codex CLI",
        available=False,
        message=error.message,
        error_code=error.code,
        retryable=error.retryable,
        suggestion=error.suggestion,
        auth_required=error.code == "auth_error",
        login_supported=error.code != "codex_not_found",
        local_home=local_home,
    )


def _url_opener(settings: CodexExecutionSettings):  # noqa: ANN202
    proxy = str(getattr(settings, "proxy", "") or "").strip()
    if proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener()


def _interactive_login_cmd(base_cmd: List[str], login_args: List[str]) -> List[str]:
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


def _new_console_creationflags() -> int:
    if os.name == "nt":
        return int(getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    return 0


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
    if any(
        marker in lower
        for marker in (
            "login",
            "auth",
            "unauthorized",
            "forbidden",
            "api key",
            "not logged in",
            "logged out",
            "not authenticated",
            "401",
            "403",
        )
    ):
        return AssistantProviderError(
            code="auth_error",
            message=text,
            retryable=False,
            suggestion="Authorize the local Codex profile or check API credentials and try again.",
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
