from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

PREFERRED_GGUF_QUANTS = (
    'Q4_K_M',
    'Q4_K_L',
    'Q5_K_M',
    'Q4_K_S',
    'Q8_0',
    'IQ4_XS',
)


@dataclass(frozen=True, kw_only=True)
class LlmSource:
    filename: str
    folder: str
    repo_id: str = ''
    url: str = ''


def parse_llm_source(raw: str) -> LlmSource:
    text = str(raw or '').strip()
    if not text:
        raise ValueError('Language model source is empty')
    parsed = urlparse(text)
    if parsed.scheme in {'http', 'https'} and parsed.netloc.lower().endswith(
        'huggingface.co'
    ):
        return _huggingface_url_source(parsed)
    if parsed.scheme in {'http', 'https'}:
        filename = _safe_filename(Path(unquote(parsed.path)).name)
        if not filename.lower().endswith('.gguf'):
            raise ValueError(
                'Direct language model URL must point to a .gguf file'
            )
        return LlmSource(
            filename=filename, folder=filename.removesuffix('.gguf'), url=text
        )

    repo_id, filename = _repo_and_filename(text)
    return LlmSource(
        filename=_safe_filename(filename) if filename else '',
        folder=_safe_folder(repo_id),
        repo_id=repo_id,
    )


def _huggingface_url_source(parsed) -> LlmSource:  # noqa: ANN001
    parts = [
        unquote(part) for part in parsed.path.strip('/').split('/') if part
    ]
    if len(parts) < 2:
        raise ValueError('Hugging Face URL must include a repo')
    repo_id = f'{parts[0]}/{parts[1]}'
    if len(parts) == 2 or (len(parts) >= 3 and parts[2] == 'tree'):
        return LlmSource(
            filename='', folder=_safe_folder(repo_id), repo_id=repo_id
        )
    if len(parts) < 5 or parts[2] not in {'resolve', 'blob'}:
        raise ValueError(
            'Hugging Face URL must include a repo or .gguf file path'
        )
    filename = '/'.join(parts[4:])
    if not filename.lower().endswith('.gguf'):
        raise ValueError('Hugging Face language model file must be .gguf')
    return LlmSource(
        filename=filename, folder=_safe_folder(repo_id), repo_id=repo_id
    )


def _repo_and_filename(text: str) -> tuple[str, str]:
    compact = re.sub(r'\s+', '/', text.strip())
    parts = [part for part in compact.replace('\\', '/').split('/') if part]
    if len(parts) < 2:
        raise ValueError(
            'Use repo or repo/file.gguf, for example org/model or org/model/model-Q4_K_M.gguf'
        )
    repo_id = f'{parts[0]}/{parts[1]}'
    if len(parts) == 2:
        return repo_id, ''
    filename = '/'.join(parts[2:])
    if not filename.lower().endswith('.gguf'):
        raise ValueError('Language model file must be .gguf')
    return repo_id, filename


def _choose_gguf_file(files: list[str]) -> str:
    ggufs = [name for name in files if name.lower().endswith('.gguf')]
    if not ggufs:
        raise ValueError('Hugging Face repo does not contain .gguf files')
    for quant in PREFERRED_GGUF_QUANTS:
        for name in ggufs:
            if quant.lower() in Path(name).name.lower():
                return name
    return sorted(ggufs, key=lambda value: value.lower())[0]


def _safe_filename(value: str) -> str:
    text = str(value or '').strip().replace('\\', '/')
    if '/' in text:
        return '/'.join(
            _safe_filename(part) for part in text.split('/') if part
        )
    if not text or text in {'.', '..'}:
        raise ValueError('Invalid language model filename')
    return re.sub(r'[^A-Za-z0-9._ -]+', '_', text)


def _safe_folder(value: str) -> str:
    text = str(value or '').strip().replace('\\', '/')
    return re.sub(r'[^A-Za-z0-9._-]+', '_', text).strip('._-') or 'model'
