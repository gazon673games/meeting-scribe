from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from application.model_download_parts.constants import BUILTIN_MODEL_REPOS, BUILTIN_MODELS, HF_REPO_SEGMENT_RE


def normalize_model_reference(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc.lower().endswith("huggingface.co"):
        parts = [part for part in parsed.path.strip("/").split("/") if part and part not in {"tree", "resolve"}]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return BUILTIN_MODEL_REPOS.get(text, text.replace("\\", "/"))


def is_builtin_model(model_name: str) -> bool:
    return model_name in BUILTIN_MODELS


def is_valid_huggingface_repo_id(value: str) -> bool:
    text = str(value or "").strip()
    if not text or "://" in text or "\\" in text or ":" in text:
        return False
    parts = text.split("/")
    if len(parts) != 2 or any(not part for part in parts):
        return False
    return all(_is_valid_repo_segment(part) for part in parts)


def is_repo_id(value: str) -> bool:
    text = str(value or "").strip()
    return is_valid_huggingface_repo_id(text) and not Path(text).expanduser().exists()


def _is_valid_repo_segment(part: str) -> bool:
    if part in {".", ".."} or part.startswith(("-", ".")) or part.endswith(("-", ".")):
        return False
    return ".." not in part and bool(HF_REPO_SEGMENT_RE.match(part))
