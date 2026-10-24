from __future__ import annotations

import time
from typing import Any

from interface.backend_parts.system_utils import module_available
from shared.domain.mappings import first_param


def start_params_from_config(owner: Any) -> dict[str, Any]:  # noqa: ANN401
    config = owner._read_config()
    ui = config.get('ui', {}) if isinstance(config.get('ui'), dict) else {}
    asr = config.get('asr', {}) if isinstance(config.get('asr'), dict) else {}
    speaker_identity = (
        config.get('speaker_identity', {})
        if isinstance(config.get('speaker_identity'), dict)
        else {}
    )
    return {
        'asrEnabled': bool(ui.get('asr_enabled', False)),
        'language': str(ui.get('lang', '')),
        'asrMode': 'split' if int(ui.get('asr_mode', 0) or 0) == 1 else 'mix',
        'profile': str(ui.get('profile', '')),
        'model': str(ui.get('model', '')),
        'wavEnabled': bool(ui.get('wav_enabled', False)),
        'outputFile': str(ui.get('output_file', '') or ''),
        'realtimeTranscriptToFile': bool(
            ui.get('rt_transcript_to_file', False)
        ),
        'speakerIdentity': dict(speaker_identity),
        **asr,
    }


def resolve_diarization_start_params(
    owner: Any, params: dict[str, Any]
) -> dict[str, Any]:  # noqa: ANN401
    if not bool(
        first_param(
            params, 'diarizationEnabled', 'diarization_enabled', default=False
        )
    ):
        return params

    backend = (
        str(
            first_param(params, 'diarBackend', 'diar_backend', default='online')
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
        return with_default_sherpa_model(owner, params)
    if backend == 'online' and not module_available('resemblyzer'):
        return with_default_sherpa_model(owner, params)
    return params


def with_default_sherpa_model(
    owner: Any, params: dict[str, Any]
) -> dict[str, Any]:  # noqa: ANN401
    from application.diarization_model_download import (
        default_cached_diarization_model,
    )

    model = default_cached_diarization_model(
        project_root=owner.project_root,
        models_dir=owner._models_dir_from_params(kind='diarization'),
        subdir='',
    )
    path = str(model.get('path') or '') if model else ''
    if not path:
        return params
    next_params = dict(params)
    next_params['diarBackend'] = 'sherpa_onnx'
    next_params['diar_backend'] = 'sherpa_onnx'
    next_params['diarSherpaEmbeddingModelPath'] = path
    next_params['diar_sherpa_embedding_model_path'] = path
    next_params.setdefault(
        'diarSherpaProvider', str(model.get('provider') or 'cpu')
    )
    next_params.setdefault(
        'diar_sherpa_provider', str(model.get('provider') or 'cpu')
    )
    return next_params


def download_then_start(
    owner: Any, controller: Any, merged: dict[str, Any]
) -> dict[str, Any]:  # noqa: ANN401
    from application.model_download import (
        is_model_cached,
        normalize_model_reference,
    )
    from application.native_asr_models import (
        is_native_asr_model,
        native_asr_archive_url,
    )

    model_name = str(merged.get('model', '')).strip()
    if not model_name:
        return controller.start_session(merged)
    normalized_model = normalize_model_reference(model_name)
    models_dir = owner._models_dir_from_params(merged)
    if not is_model_cached(normalized_model, models_dir=models_dir):
        if (
            is_native_asr_model(model_name)
            and native_asr_archive_url(model_name) is None
        ):
            raise RuntimeError(
                f"Native ASR model '{model_name}' is not installed in {models_dir}"
            )
        controller.begin_model_download(normalized_model)
        try:
            owner.download_model(
                {
                    'name': normalized_model,
                    'modelsDir': str(models_dir),
                    'modelsSubdir': '.',
                }
            )
            wait_limit_s = float(
                first_param(
                    merged,
                    'downloadWaitTimeoutS',
                    'download_wait_timeout_s',
                    default=3600,
                )
            )
            start = time.monotonic()
            while time.monotonic() - start < wait_limit_s:
                if owner._shutdown.is_set():
                    raise RuntimeError(
                        'Session startup cancelled during shutdown'
                    )
                state = owner._download_record(normalized_model)
                if state.get('state') == 'done':
                    break
                if state.get('state') == 'error':
                    raise RuntimeError(
                        str(state.get('error') or 'ASR model download failed')
                    )
                owner._shutdown.wait(0.2)
            else:
                raise RuntimeError(
                    f"Timed out while downloading model '{normalized_model}'"
                )
        except Exception as error:
            controller.finish_model_download(str(error))
            raise
        controller.finish_model_download('')
    if owner._shutdown.is_set():
        raise RuntimeError('Session startup cancelled during shutdown')
    return controller.start_session(merged)
