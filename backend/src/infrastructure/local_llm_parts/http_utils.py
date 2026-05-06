from __future__ import annotations

import urllib.error


def http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def http_suggestion(status: int) -> str:
    if status == 404:
        return "Check the provider type, base URL, and model name."
    if status in (401, 403):
        return "Check the local endpoint API key or disable auth in the local server."
    if status == 429:
        return "The local server is busy; retry or use a smaller model."
    if status >= 500:
        return "Check the local LLM server logs."
    return "Check the selected local LLM profile."
