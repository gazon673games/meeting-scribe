from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from application.asr_language import (
    initial_prompt_for_language,
    normalize_asr_language,
    runtime_asr_language,
)
from application.asr_profiles import (
    profile_defaults,
    profile_requires_streaming,
)
from application.asr_session import ASRSessionSettings
from application.native_asr_models import NEMOTRON_MODEL_ID
from asr.domain.types import Mode
from diarization.domain.types import DiarBackend
from interface.session_controller_parts.helpers import (
    first_param,
    normalize_diar_backend,
    safe_bool,
    safe_float_clamped,
    safe_int,
    speaker_identity_params,
)


class _StreamingSettings(TypedDict):
    streaming_enabled: bool
    streaming_chunk_interval_s: float
    streaming_endpoint_silence_ms: float


class _DiarizationSettings(TypedDict):
    diarization_enabled: bool
    diar_backend: DiarBackend
    diarization_sidecar_enabled: bool
    diarization_queue_size: int
    diar_sherpa_embedding_model_path: str
    diar_sherpa_provider: str
    diar_sherpa_num_threads: int


class _IdentitySettings(TypedDict):
    speaker_identity_enabled: bool
    speaker_identity_persistent_profiles_enabled: bool
    speaker_identity_backend: str
    speaker_identity_app_data_dir: str
    speaker_identity_auto_match_threshold: float
    speaker_identity_uncertain_match_threshold: float
    speaker_identity_min_speech_ms_for_embedding: int
    speaker_identity_min_quality_score: float


def asr_settings_from_params(
    params: dict[str, Any],
    *,
    source_speaker_labels: dict[str, str] | None = None,
) -> ASRSessionSettings:
    language = normalize_asr_language(
        str(first_param(params, 'language', 'lang', default='ru'))
    )
    mode_raw = (
        str(first_param(params, 'asrMode', 'asr_mode', default='split'))
        .strip()
        .lower()
    )
    mode: Mode = 'split' if mode_raw in ('1', 'split', 'sources') else 'mix'
    overload_strategy = (
        str(
            first_param(
                params,
                'overload_strategy',
                'overloadStrategy',
                default='drop_old',
            )
        )
        .strip()
        .lower()
    )
    profile = str(
        first_param(params, 'profile', 'asr_profile', default='') or ''
    ).strip()
    model_name = str(
        first_param(params, 'model', 'model_name', default='medium') or 'medium'
    )
    streaming_settings = _streaming_settings(
        params, profile=profile, model_name=model_name
    )
    identity_params = speaker_identity_params(params)
    models_directory = str(params.get('asrModelsDir') or '').strip()
    settings = ASRSessionSettings(
        models_dir=Path(models_directory) if models_directory else None,
        language=language,
        mode=mode,
        model_name=model_name,
        device=str(params.get('device', 'cuda') or 'cuda'),
        compute_type=str(
            first_param(
                params, 'compute_type', 'computeType', default='float16'
            )
            or 'float16'
        ),
        cpu_threads=safe_int(
            first_param(params, 'cpu_threads', 'cpuThreads', default=0),
            default=0,
            lo=0,
            hi=64,
        ),
        num_workers=safe_int(
            first_param(params, 'num_workers', 'numWorkers', default=1),
            default=1,
            lo=1,
            hi=16,
        ),
        beam_size=safe_int(
            first_param(params, 'beam_size', 'beamSize', default=5),
            default=5,
            lo=1,
            hi=20,
        ),
        endpoint_silence_ms=safe_float_clamped(
            first_param(
                params,
                'endpoint_silence_ms',
                'endpointSilenceMs',
                default=650.0,
            ),
            default=650.0,
            lo=50.0,
            hi=5000.0,
        ),
        max_segment_s=safe_float_clamped(
            first_param(params, 'max_segment_s', 'maxSegmentS', default=7.0),
            default=7.0,
            lo=1.0,
            hi=60.0,
        ),
        overlap_ms=safe_float_clamped(
            first_param(params, 'overlap_ms', 'overlapMs', default=200.0),
            default=200.0,
            lo=0.0,
            hi=2000.0,
        ),
        vad_energy_threshold=safe_float_clamped(
            first_param(
                params,
                'vad_energy_threshold',
                'vadEnergyThreshold',
                default=0.0055,
            ),
            default=0.0055,
            lo=1e-5,
            hi=1.0,
        ),
        overload_strategy='keep_all'
        if overload_strategy == 'keep_all'
        else 'drop_old',
        overload_enter_qsize=safe_int(
            first_param(
                params, 'overload_enter_qsize', 'overloadEnterQsize', default=18
            ),
            default=18,
            lo=1,
            hi=999,
        ),
        overload_exit_qsize=safe_int(
            first_param(
                params, 'overload_exit_qsize', 'overloadExitQsize', default=6
            ),
            default=6,
            lo=1,
            hi=999,
        ),
        overload_hard_qsize=safe_int(
            first_param(
                params, 'overload_hard_qsize', 'overloadHardQsize', default=28
            ),
            default=28,
            lo=1,
            hi=999,
        ),
        overload_beam_cap=safe_int(
            first_param(
                params, 'overload_beam_cap', 'overloadBeamCap', default=2
            ),
            default=2,
            lo=1,
            hi=20,
        ),
        overload_max_segment_s=safe_float_clamped(
            first_param(
                params,
                'overload_max_segment_s',
                'overloadMaxSegmentS',
                default=5.0,
            ),
            default=5.0,
            lo=0.5,
            hi=60.0,
        ),
        overload_overlap_ms=safe_float_clamped(
            first_param(
                params,
                'overload_overlap_ms',
                'overloadOverlapMs',
                default=120.0,
            ),
            default=120.0,
            lo=0.0,
            hi=2000.0,
        ),
        asr_language=runtime_asr_language(language),
        asr_initial_prompt=initial_prompt_for_language(language),
        asr_hotwords=str(
            first_param(params, 'hotwords', 'asr_hotwords', default='') or ''
        ).strip()
        or None,
        source_speaker_labels=dict(source_speaker_labels or {}),
        **_diarization_settings(params),
        **streaming_settings,
        **_speaker_identity_settings(identity_params),
    )
    if (
        not settings.overload_exit_qsize
        < settings.overload_enter_qsize
        <= settings.overload_hard_qsize
    ):
        raise ValueError(
            'Overload thresholds must satisfy exit < enter <= hard'
        )
    if (
        settings.speaker_identity_uncertain_match_threshold
        > settings.speaker_identity_auto_match_threshold
    ):
        raise ValueError(
            'Uncertain speaker threshold must not exceed the automatic match threshold'
        )
    return settings


def _streaming_settings(
    params: dict[str, Any], *, profile: str, model_name: str
) -> _StreamingSettings:
    streaming_required = profile_requires_streaming(profile)
    native_streaming_required = model_name == NEMOTRON_MODEL_ID
    streaming_defaults = profile_defaults(profile) if streaming_required else {}
    streaming_chunk_default = float(
        streaming_defaults.get('streaming_chunk_interval_s', 1.0)
    )
    if native_streaming_required:
        streaming_chunk_default = min(streaming_chunk_default, 0.56)
    streaming_endpoint_default = float(
        streaming_defaults.get('streaming_endpoint_silence_ms', 300.0)
    )
    streaming_enabled = (
        safe_bool(
            first_param(
                params,
                'streamingEnabled',
                'streaming_enabled',
                default=streaming_defaults.get('streaming_enabled', False),
            ),
            default=bool(streaming_defaults.get('streaming_enabled', False)),
        )
        or streaming_required
        or native_streaming_required
    )
    streaming_chunk_interval_s = safe_float_clamped(
        streaming_chunk_default
        if streaming_required
        else first_param(
            params,
            'streamingChunkIntervalS',
            'streaming_chunk_interval_s',
            default=streaming_chunk_default,
        ),
        default=streaming_chunk_default,
        lo=0.1,
        hi=5.0,
    )
    streaming_endpoint_silence_ms = safe_float_clamped(
        streaming_endpoint_default
        if streaming_required
        else first_param(
            params,
            'streamingEndpointSilenceMs',
            'streaming_endpoint_silence_ms',
            default=streaming_endpoint_default,
        ),
        default=streaming_endpoint_default,
        lo=50.0,
        hi=5000.0,
    )
    return {
        'streaming_enabled': streaming_enabled,
        'streaming_chunk_interval_s': streaming_chunk_interval_s,
        'streaming_endpoint_silence_ms': streaming_endpoint_silence_ms,
    }


def _diarization_settings(params: dict[str, Any]) -> _DiarizationSettings:
    return {
        'diarization_enabled': bool(
            first_param(
                params,
                'diarizationEnabled',
                'diarization_enabled',
                default=False,
            )
        ),
        'diar_backend': normalize_diar_backend(
            first_param(params, 'diarBackend', 'diar_backend', default='online')
        ),
        'diarization_sidecar_enabled': bool(
            first_param(
                params,
                'diarizationSidecarEnabled',
                'diarization_sidecar_enabled',
                default=True,
            )
        ),
        'diarization_queue_size': safe_int(
            first_param(
                params,
                'diarization_queue_size',
                'diarizationQueueSize',
                default=50,
            ),
            default=50,
            lo=1,
            hi=500,
        ),
        'diar_sherpa_embedding_model_path': str(
            first_param(
                params,
                'diarSherpaEmbeddingModelPath',
                'diar_sherpa_embedding_model_path',
                'diarizationSherpaEmbeddingModelPath',
                default='',
            )
            or ''
        ).strip(),
        'diar_sherpa_provider': str(
            first_param(
                params,
                'diarSherpaProvider',
                'diar_sherpa_provider',
                default='cpu',
            )
            or 'cpu'
        ).strip()
        or 'cpu',
        'diar_sherpa_num_threads': safe_int(
            first_param(
                params,
                'diarSherpaNumThreads',
                'diar_sherpa_num_threads',
                default=1,
            ),
            default=1,
            lo=1,
            hi=32,
        ),
    }


def _speaker_identity_settings(
    identity_params: dict[str, Any],
) -> _IdentitySettings:
    return {
        'speaker_identity_enabled': safe_bool(
            identity_params.get('enabled', False), default=False
        ),
        'speaker_identity_persistent_profiles_enabled': safe_bool(
            identity_params.get('persistent_profiles_enabled', False),
            default=False,
        ),
        'speaker_identity_backend': str(
            identity_params.get('backend', 'file') or 'file'
        ).strip()
        or 'file',
        'speaker_identity_app_data_dir': str(
            identity_params.get('app_data_dir', '') or ''
        ).strip(),
        'speaker_identity_auto_match_threshold': safe_float_clamped(
            identity_params.get('auto_match_threshold', 0.84),
            default=0.84,
            lo=0.0,
            hi=1.0,
        ),
        'speaker_identity_uncertain_match_threshold': safe_float_clamped(
            identity_params.get('uncertain_match_threshold', 0.74),
            default=0.74,
            lo=0.0,
            hi=1.0,
        ),
        'speaker_identity_min_speech_ms_for_embedding': safe_int(
            identity_params.get('min_speech_ms_for_embedding', 8000),
            default=8000,
            lo=0,
            hi=10000000,
        ),
        'speaker_identity_min_quality_score': safe_float_clamped(
            identity_params.get('min_quality_score', 0.65),
            default=0.65,
            lo=0.0,
            hi=1.0,
        ),
    }
