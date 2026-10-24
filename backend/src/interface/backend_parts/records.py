from __future__ import annotations

from typing import TypedDict


class DownloadState(TypedDict, total=False):
    state: str
    message: str
    error: str
    downloadedBytes: int
    totalBytes: int
    speedBps: float
    proxy: bool
    path: str
