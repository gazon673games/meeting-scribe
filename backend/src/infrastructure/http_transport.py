from __future__ import annotations

from http.client import HTTPResponse
from urllib.request import ProxyHandler, Request, build_opener, urlopen


def open_url(request: Request, *, proxy: str, timeout: float) -> HTTPResponse:
    proxy = str(proxy or '').strip()
    if not proxy:
        return urlopen(request, timeout=timeout)
    opener = build_opener(ProxyHandler({'http': proxy, 'https': proxy}))
    return opener.open(request, timeout=timeout)
