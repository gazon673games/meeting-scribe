from __future__ import annotations

import urllib.error

_HTTP_SUGGESTIONS = {
    404: 'Check the provider type, base URL, and model name.',
    401: 'Check the local endpoint API key or disable auth in the local server.',
    403: 'Check the local endpoint API key or disable auth in the local server.',
    429: 'The local server is busy; retry or use a smaller model.',
}


def http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode('utf-8', errors='replace').strip()
    except (OSError, ValueError):
        return ''


def http_suggestion(status: int) -> str:
    suggestion = _HTTP_SUGGESTIONS.get(status)
    if suggestion is not None:
        return suggestion
    if status >= 500:
        return 'Check the local LLM server logs.'
    return 'Check the selected local LLM profile.'
