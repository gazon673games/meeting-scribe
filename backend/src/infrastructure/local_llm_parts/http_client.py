from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from infrastructure.local_llm_parts.errors import LocalLlmError
from infrastructure.local_llm_parts.http_utils import http_error_body, http_suggestion


def http_open(request: urllib.request.Request, timeout_s: int) -> Any:  # noqa: ANN401
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=max(1, int(timeout_s or 1)))
    except urllib.error.HTTPError as error:
        status = int(error.code or 0)
        body = http_error_body(error)
        raise LocalLlmError(
            code="local_llm_http_error",
            message=f"Local LLM endpoint returned HTTP {status}: {body or getattr(error, 'reason', '')}",
            retryable=status >= 500 or status == 429,
            suggestion=http_suggestion(status),
            status_code=status,
        ) from error
    except (TimeoutError, urllib.error.URLError, OSError) as error:
        raise LocalLlmError(
            code="local_llm_unavailable",
            message=f"{type(error).__name__}: {error}",
            suggestion="Start the local LLM server and check the profile base URL.",
        ) from error


def request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: int,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = {"Accept": "application/json", **dict(headers or {})}
    if data is not None:
        request_headers["Content-Type"] = "application/json"

    response = http_open(
        urllib.request.Request(url, data=data, headers=request_headers, method=method),
        timeout_s,
    )
    try:
        raw = response.read().decode("utf-8", errors="replace")
    finally:
        response.close()
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LocalLlmError(
            code="local_llm_bad_response",
            message=f"Local LLM endpoint returned invalid JSON: {error}",
            suggestion="Check that the selected provider type matches the local server API.",
        ) from error
    return parsed if isinstance(parsed, dict) else {"data": parsed}
