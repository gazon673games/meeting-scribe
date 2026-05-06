from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from application.codex_assistant import CodexAssistantRequest, CodexAssistantResult, CodexExecutionSettings
from application.codex_config import CodexProfile
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
from infrastructure.codex_cli_parts import (
    CodexCommandResolver,
    build_exec_cmd,
    classify_codex_error,
    codex_home,
    codex_not_found_error,
    interactive_login_cmd,
    new_console_creationflags,
    process_output_text,
    provider_info_from_error,
    read_output_file,
    settings_project_root,
    status_error_text,
)

_OPENAI_API_PING_URL = "https://api.openai.com/v1/models"


@dataclass(frozen=True)
class _RunContext:
    profile: CodexProfile
    settings: CodexExecutionSettings
    tmp_dir: Path
    base_cmd: List[str]
    source: str
    env: dict


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
        project_root = settings_project_root(settings)
        cache_key = self._status_key(settings, project_root)
        cached = self._cached_status(cache_key)
        if cached is not None:
            return cached

        local_home = str(codex_home(project_root))
        resolved = self._resolver.resolve(settings)
        if resolved is None:
            return self._cache_status(cache_key, self._info_from_error(codex_not_found_error(), local_home=local_home))

        base_cmd, source = resolved
        status_check = self._run_status_check(settings, project_root, base_cmd, source)
        if isinstance(status_check, AssistantProviderError):
            info = self._info_from_error(status_check, local_home=local_home)
            return self._cache_status(cache_key, info)
        returncode, status_text = status_check
        if returncode == 0:
            info = self._status_ok_info(source=source, status_text=status_text, local_home=local_home)
            return self._cache_status(cache_key, info)
        info = self._status_failed_info(source=source, returncode=returncode, status_text=status_text, local_home=local_home)
        return self._cache_status(cache_key, info)

    def start_login(
        self,
        settings: CodexExecutionSettings,
        *,
        device_auth: bool = False,
    ) -> AssistantProviderLoginResult:
        project_root = settings_project_root(settings)
        local_home = str(codex_home(project_root))
        resolved = self._resolver.resolve(settings)
        self._clear_status_cache()
        if resolved is None:
            error = codex_not_found_error()
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
        cmd = interactive_login_cmd(list(base_cmd), login_args)
        subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=self._build_env(settings, project_root, tmp_dir),
            creationflags=new_console_creationflags(),
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
        started_at = time.time()
        out_path: Optional[Path] = None
        try:
            prepared = self._prepare_run_context(request, started_at)
            if isinstance(prepared, CodexAssistantResult):
                return prepared
            context = prepared
            out_path = context.tmp_dir / f"codex_last_{os.getpid()}_{uuid.uuid4().hex}.txt"
            cmd = build_exec_cmd(list(context.base_cmd), context.profile, out_path)
            proc = self._run_exec_process(request, context, cmd)
            out_text = read_output_file(out_path, proc)
            if proc.returncode != 0:
                err = (proc.stderr or "").strip() or (proc.stdout or "").strip()
                err = f"{err}\n(source={context.source})" if err else f"codex exec failed with code {proc.returncode}"
                return self._error_result(context.profile, request.original_cmd, classify_codex_error(err), started_at)
            return self._result(True, context.profile, request.original_cmd, out_text or "(empty response)", started_at)
        except subprocess.TimeoutExpired:
            return self._error_result(
                request.profile,
                request.original_cmd,
                AssistantProviderError(
                    code="timeout",
                    message=f"timeout after {int(request.settings.timeout_s)}s",
                    retryable=True,
                    suggestion="Increase assistant timeout or use a faster model/profile.",
                ),
                started_at,
            )
        except FileNotFoundError:
            return self._error_result(request.profile, request.original_cmd, codex_not_found_error(runtime=True), started_at)
        except Exception as exc:
            return self._error_result(
                request.profile,
                request.original_cmd,
                AssistantProviderError(
                    code="unknown_error",
                    message=f"{type(exc).__name__}: {exc}",
                    retryable=True,
                ),
                started_at,
            )
        finally:
            if out_path is not None:
                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _run_status_check(
        self,
        settings: CodexExecutionSettings,
        project_root: Path,
        base_cmd: List[str],
        source: str,
    ) -> tuple[int, str] | AssistantProviderError:
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
            return proc.returncode, process_output_text(proc)
        except subprocess.TimeoutExpired:
            return AssistantProviderError(
                code="provider_status_timeout",
                message=f"Codex CLI was found via {source}, but login status timed out.",
                retryable=True,
                suggestion="Try again or run codex login status manually in a terminal.",
            )
        except FileNotFoundError:
            return codex_not_found_error(runtime=True)

    def _status_ok_info(self, *, source: str, status_text: str, local_home: str) -> AssistantProviderInfo:
        detail = status_text or "Codex login is ready"
        return AssistantProviderInfo(
            id=self.provider_id,
            label=self.provider_label,
            available=True,
            message=f"Found via {source}; {detail}",
            login_supported=True,
            local_home=local_home,
        )

    def _status_failed_info(
        self,
        *,
        source: str,
        returncode: int,
        status_text: str,
        local_home: str,
    ) -> AssistantProviderInfo:
        error = classify_codex_error(status_error_text(status_text, source=source, returncode=returncode))
        suggestion = "Authorize the local Codex profile before using AI Assistant." if error.code == "auth_error" else error.suggestion
        return AssistantProviderInfo(
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
        )

    def _prepare_run_context(self, request: CodexAssistantRequest, started_at: float) -> _RunContext | CodexAssistantResult:
        profile = request.profile
        settings = request.settings
        tmp_dir = project_runtime_dir(request.project_root, "codex")
        env = self._build_env(settings, request.project_root, tmp_dir)
        resolved = self._resolver.resolve(settings)
        if resolved is None:
            return self._error_result(profile, request.original_cmd, codex_not_found_error(), started_at)
        base_cmd, source = resolved
        return _RunContext(
            profile=profile,
            settings=settings,
            tmp_dir=tmp_dir,
            base_cmd=list(base_cmd),
            source=source,
            env=env,
        )

    def _run_exec_process(self, request: CodexAssistantRequest, context: _RunContext, cmd: List[str]) -> subprocess.CompletedProcess:
        prompt = request.prompt.encode("utf-8", errors="replace").decode("utf-8")
        return subprocess.run(
            cmd,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            cwd=str(request.project_root),
            env=context.env,
            timeout=max(10, int(context.settings.timeout_s)),
        )

    @staticmethod
    def _build_env(settings: CodexExecutionSettings, project_root: Path, tmp_dir: Path) -> dict:
        env = os.environ.copy()
        for key in ("TMP", "TEMP", "TMPDIR"):
            env[key] = str(tmp_dir)
        local_home = codex_home(Path(project_root))
        local_home.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(local_home)
        proxy = str(settings.proxy or "").strip()
        if proxy:
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
                env[key] = proxy
        return env

    def _status_key(self, settings: CodexExecutionSettings, project_root: Path) -> tuple:
        return (
            tuple(str(item) for item in settings.command_tokens),
            tuple(str(item) for item in settings.path_hints),
            str(settings.proxy or ""),
            str(project_root.resolve()),
        )

    def _cached_status(self, cache_key: tuple) -> Optional[AssistantProviderInfo]:
        if self._status_cache_key != cache_key or self._status_cache_info is None:
            return None
        if time.monotonic() - self._status_cache_ts > self._status_ttl_s:
            return None
        return self._status_cache_info

    def _cache_status(self, cache_key: tuple, info: AssistantProviderInfo) -> AssistantProviderInfo:
        self._status_cache_key = cache_key
        self._status_cache_ts = time.monotonic()
        self._status_cache_info = info
        return info

    def _clear_status_cache(self) -> None:
        self._status_cache_key = None
        self._status_cache_info = None

    def _info_from_error(self, error: AssistantProviderError, *, local_home: str = "") -> AssistantProviderInfo:
        return provider_info_from_error(
            error,
            provider_id=self.provider_id,
            provider_label=self.provider_label,
            local_home=local_home,
        )

    @staticmethod
    def _result(ok: bool, profile: CodexProfile, original_cmd: str, text: str, started_at: float) -> CodexAssistantResult:
        return CodexAssistantResult(
            ok=bool(ok),
            profile=str(profile.label),
            cmd=str(original_cmd),
            text=str(text),
            dt_s=time.time() - started_at,
            provider=ASSISTANT_PROVIDER_CODEX,
            model=str(profile.model or ""),
        )

    @staticmethod
    def _error_result(
        profile: CodexProfile,
        original_cmd: str,
        error: AssistantProviderError,
        started_at: float,
    ) -> CodexAssistantResult:
        return result_from_error(
            profile=profile,
            cmd=original_cmd,
            provider=ASSISTANT_PROVIDER_CODEX,
            model=str(profile.model or ""),
            error=error,
            started_at=started_at,
        )


def _url_opener(settings: CodexExecutionSettings):  # noqa: ANN202
    proxy = str(getattr(settings, "proxy", "") or "").strip()
    if proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener()
