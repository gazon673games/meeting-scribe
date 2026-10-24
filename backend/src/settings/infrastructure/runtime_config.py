from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from settings.application.config_repository import ConfigRepository

_logger = logging.getLogger(__name__)


def _merge_bundled_codex_config(
    repository: ConfigRepository, current: dict, bundled: dict
) -> None:
    current_codex = current.get('codex', {})
    bundled_codex = bundled.get('codex', {})
    if not isinstance(current_codex, dict) or not isinstance(
        bundled_codex, dict
    ):
        return
    if 'codex' in current and bool(current_codex.get('profiles')):
        return
    merged = {**current, 'codex': bundled_codex}
    try:
        repository.write(merged)
    except (OSError, UnicodeError):
        _logger.warning(
            'Could not save bundled assistant configuration',
            exc_info=True,
        )


def ensure_runtime_config(
    project_root: Path, repository: ConfigRepository
) -> None:
    if not getattr(sys, 'frozen', False):
        return

    bundled_config = (
        Path(getattr(sys, '_MEIPASS', project_root)) / 'config.json'
    )
    if not bundled_config.exists():
        return

    if not repository.exists():
        try:
            bundled = json.loads(bundled_config.read_text(encoding='utf-8'))
            if not isinstance(bundled, dict):
                _logger.warning(
                    'Bundled configuration root must be an object: %s',
                    bundled_config,
                )
                return
            repository.write(bundled)
        except (OSError, UnicodeError, json.JSONDecodeError):
            _logger.warning(
                'Could not initialize runtime configuration from %s',
                bundled_config,
                exc_info=True,
            )
        return

    try:
        current = repository.read()
        bundled = json.loads(bundled_config.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _logger.warning(
            'Could not read runtime or bundled configuration from %s',
            bundled_config,
            exc_info=True,
        )
        return

    if not isinstance(current, dict) or not isinstance(bundled, dict):
        _logger.warning(
            'Runtime and bundled configuration roots must be objects'
        )
        return

    _merge_bundled_codex_config(repository, current, bundled)
