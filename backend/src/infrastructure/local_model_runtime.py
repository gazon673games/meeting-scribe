from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from application.codex_config import CodexProfile
from assistant.application.local_models import ModelStatusSink

_logger = logging.getLogger(__name__)


def start(
    profile: CodexProfile, project_root: Path, emit: ModelStatusSink
) -> dict[str, Any]:
    provider_id = profile.provider_id.strip().lower()
    if provider_id == 'local':
        from infrastructure.llama_cpp_runner import load_model_async

        return load_model_async(profile, project_root, emit)
    if provider_id != 'openai_local':
        raise ValueError(
            f'Provider {provider_id} manages its own model lifecycle'
        )
    from infrastructure.local_llm import start_local_llm_async

    return start_local_llm_async(profile, project_root, emit)


def stop(profile: CodexProfile) -> dict[str, Any]:
    provider_id = profile.provider_id.strip().lower()
    if provider_id == 'local':
        from infrastructure.llama_cpp_runner import unload_model

        return unload_model(profile)
    if provider_id != 'openai_local':
        raise ValueError(
            f'Provider {provider_id} manages its own model lifecycle'
        )
    from infrastructure.local_llm import stop_local_llm

    return stop_local_llm(profile)


def shutdown() -> None:
    from infrastructure.llama_cpp_runner import shutdown as close_models
    from infrastructure.local_llm_parts.runtime_server import (
        shutdown as close_servers,
    )

    failures: list[BaseException] = []
    for close in (close_models, close_servers):
        try:
            close()
        except BaseException as error:
            failures.append(error)
            _logger.exception('Local model cleanup failed')
    if failures:
        raise failures[0]
