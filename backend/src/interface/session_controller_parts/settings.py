from __future__ import annotations

from typing import Any, Dict, Optional

from application.asr_language import initial_prompt_for_language, normalize_asr_language, runtime_asr_language
from application.asr_profiles import profile_defaults, profile_requires_streaming
from application.asr_session import ASRSessionSettings
from interface.session_controller_parts.helpers import (
    first_param,
    normalize_diar_backend,
    safe_bool,
    safe_float_clamped,
    safe_int,
    speaker_identity_params,
)


def asr_settings_from_params(
    params: Dict[str, Any],
    *,
    source_speaker_labels: Optional[Dict[str, str]] = None,
) -> ASRSessionSettings:
    language = normalize_asr_language(str(params.get("language", params.get("lang", "ru"))))
    mode_raw = str(params.get("asrMode", params.get("asr_mode", "split"))).strip().lower()
    mode = "split" if mode_raw in {"1", "split", "sources"} else "mix"
    overload_strategy = str(params.get("overload_strategy", params.get("overloadStrategy", "drop_old"))).strip().lower()
    profile = str(params.get("profile", params.get("asr_profile", "")) or "").strip()
    streaming_required = profile_requires_streaming(profile)
    streaming_defaults = profile_defaults(profile) if streaming_required else {}
    streaming_chunk_default = float(streaming_defaults.get("streaming_chunk_interval_s", 1.0))
    streaming_endpoint_default = float(streaming_defaults.get("streaming_endpoint_silence_ms", 300.0))
    streaming_enabled = (
        safe_bool(
            first_param(
                params,
                "streamingEnabled",
                "streaming_enabled",
                default=streaming_defaults.get("streaming_enabled", False),
            ),
            bool(streaming_defaults.get("streaming_enabled", False)),
        )
        or streaming_required
    )
    streaming_chunk_interval_s = safe_float_clamped(
        streaming_chunk_default
        if streaming_required
        else first_param(
            params,
            "streamingChunkIntervalS",
            "streaming_chunk_interval_s",
            default=streaming_chunk_default,
        ),
        streaming_chunk_default,
        0.1,
        5.0,
    )
    streaming_endpoint_silence_ms = safe_float_clamped(
        streaming_endpoint_default
        if streaming_required
        else first_param(
            params,
            "streamingEndpointSilenceMs",
            "streaming_endpoint_silence_ms",
            default=streaming_endpoint_default,
        ),
        streaming_endpoint_default,
        50.0,
        5000.0,
    )
    identity_params = speaker_identity_params(params)
    return ASRSessionSettings(
        language=language,
        mode=mode,  # type: ignore[arg-type]
        model_name=str(params.get("model", params.get("model_name", "medium")) or "medium"),
        device=str(params.get("device", "cuda") or "cuda"),
        compute_type=str(params.get("compute_type", params.get("computeType", "float16")) or "float16"),
        cpu_threads=safe_int(params.get("cpu_threads", params.get("cpuThreads", 0)), 0, 0, 64),
        num_workers=safe_int(params.get("num_workers", params.get("numWorkers", 1)), 1, 1, 16),
        beam_size=safe_int(params.get("beam_size", params.get("beamSize", 5)), 5, 1, 20),
        endpoint_silence_ms=safe_float_clamped(
            params.get("endpoint_silence_ms", params.get("endpointSilenceMs", 650.0)),
            650.0,
            50.0,
            5000.0,
        ),
        max_segment_s=safe_float_clamped(params.get("max_segment_s", params.get("maxSegmentS", 7.0)), 7.0, 1.0, 60.0),
        overlap_ms=safe_float_clamped(params.get("overlap_ms", params.get("overlapMs", 200.0)), 200.0, 0.0, 2000.0),
        vad_energy_threshold=safe_float_clamped(
            params.get("vad_energy_threshold", params.get("vadEnergyThreshold", 0.0055)),
            0.0055,
            1e-5,
            1.0,
        ),
        overload_strategy="keep_all" if overload_strategy == "keep_all" else "drop_old",  # type: ignore[arg-type]
        overload_enter_qsize=safe_int(params.get("overload_enter_qsize", params.get("overloadEnterQsize", 18)), 18, 1, 999),
        overload_exit_qsize=safe_int(params.get("overload_exit_qsize", params.get("overloadExitQsize", 6)), 6, 1, 999),
        overload_hard_qsize=safe_int(params.get("overload_hard_qsize", params.get("overloadHardQsize", 28)), 28, 1, 999),
        overload_beam_cap=safe_int(params.get("overload_beam_cap", params.get("overloadBeamCap", 2)), 2, 1, 20),
        overload_max_segment_s=safe_float_clamped(
            params.get("overload_max_segment_s", params.get("overloadMaxSegmentS", 5.0)),
            5.0,
            0.5,
            60.0,
        ),
        overload_overlap_ms=safe_float_clamped(
            params.get("overload_overlap_ms", params.get("overloadOverlapMs", 120.0)),
            120.0,
            0.0,
            2000.0,
        ),
        asr_language=runtime_asr_language(language),
        asr_initial_prompt=initial_prompt_for_language(language),
        source_speaker_labels=dict(source_speaker_labels or {}),
        diarization_enabled=bool(params.get("diarizationEnabled", params.get("diarization_enabled", False))),
        diar_backend=normalize_diar_backend(params.get("diarBackend", params.get("diar_backend", "online"))),
        diarization_sidecar_enabled=bool(
            params.get("diarizationSidecarEnabled", params.get("diarization_sidecar_enabled", True))
        ),
        diarization_queue_size=safe_int(
            params.get("diarization_queue_size", params.get("diarizationQueueSize", 50)),
            50,
            1,
            500,
        ),
        diar_sherpa_embedding_model_path=str(
            first_param(
                params,
                "diarSherpaEmbeddingModelPath",
                "diar_sherpa_embedding_model_path",
                "diarizationSherpaEmbeddingModelPath",
                default="",
            )
            or ""
        ).strip(),
        diar_sherpa_provider=str(
            first_param(params, "diarSherpaProvider", "diar_sherpa_provider", default="cpu") or "cpu"
        ).strip()
        or "cpu",
        diar_sherpa_num_threads=safe_int(
            first_param(params, "diarSherpaNumThreads", "diar_sherpa_num_threads", default=1),
            1,
            1,
            32,
        ),
        streaming_enabled=streaming_enabled,
        streaming_chunk_interval_s=streaming_chunk_interval_s,
        streaming_endpoint_silence_ms=streaming_endpoint_silence_ms,
        speaker_identity_enabled=safe_bool(identity_params.get("enabled", False), False),
        speaker_identity_persistent_profiles_enabled=safe_bool(
            identity_params.get("persistent_profiles_enabled", False),
            False,
        ),
        speaker_identity_backend=str(identity_params.get("backend", "file") or "file").strip() or "file",
        speaker_identity_app_data_dir=str(identity_params.get("app_data_dir", "") or "").strip(),
        speaker_identity_auto_match_threshold=safe_float_clamped(
            identity_params.get("auto_match_threshold", 0.84),
            0.84,
            0.0,
            1.0,
        ),
        speaker_identity_uncertain_match_threshold=safe_float_clamped(
            identity_params.get("uncertain_match_threshold", 0.74),
            0.74,
            0.0,
            1.0,
        ),
        speaker_identity_min_speech_ms_for_embedding=safe_int(
            identity_params.get("min_speech_ms_for_embedding", 8000),
            8000,
            0,
            10_000_000,
        ),
        speaker_identity_min_quality_score=safe_float_clamped(
            identity_params.get("min_quality_score", 0.65),
            0.65,
            0.0,
            1.0,
        ),
    )

