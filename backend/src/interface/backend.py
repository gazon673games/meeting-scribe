from __future__ import annotations

import time
from typing import Any

from interface import backend_impl as _impl
from interface.backend_impl import ElectronBackend as _ElectronBackendImpl
from interface.backend_parts.session_orchestration import (
    with_default_sherpa_model,
)
from shared.domain.mappings import first_param


def _module_available(name: str) -> bool:
    return _impl._module_available(name)


def _nvidia_gpu_snapshot() -> list[dict[str, Any]]:
    return _impl._nvidia_gpu_snapshot()


class ElectronBackend(_ElectronBackendImpl):
    def _resolve_diarization_start_params(
        self, params: dict[str, Any]
    ) -> dict[str, Any]:
        if not bool(
            first_param(
                params,
                'diarizationEnabled',
                'diarization_enabled',
                default=False,
            )
        ):
            return params

        backend = (
            str(
                first_param(
                    params, 'diarBackend', 'diar_backend', default='online'
                )
                or 'online'
            )
            .strip()
            .lower()
        )
        sherpa_path = str(
            first_param(
                params,
                'diarSherpaEmbeddingModelPath',
                'diar_sherpa_embedding_model_path',
                default='',
            )
            or ''
        ).strip()

        if backend == 'sherpa_onnx' and not sherpa_path:
            return with_default_sherpa_model(self, params)
        if backend == 'online' and not _module_available('resemblyzer'):
            return with_default_sherpa_model(self, params)
        return params

    def _gpu_snapshot(self, *, max_age_s: float) -> list[dict[str, Any]]:
        with self._resource_lock:
            now = time.monotonic()
            if self._gpu_cache_ts > 0.0 and now - self._gpu_cache_ts < max(
                0.0, float(max_age_s)
            ):
                return [dict(gpu) for gpu in self._gpu_cache]
            gpus = _nvidia_gpu_snapshot()
            self._gpu_cache = [dict(gpu) for gpu in gpus]
            self._gpu_cache_ts = now
            return [dict(gpu) for gpu in self._gpu_cache]


__all__ = ['ElectronBackend']
