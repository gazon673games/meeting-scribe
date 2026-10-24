from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from application.asr_profiles import (
    PROFILE_BALANCED,
    PROFILE_CUSTOM,
    PROFILE_QUALITY,
    PROFILE_REALTIME,
    PROFILE_ULTRA_FAST,
)
from application.codex_config import CodexProfile
from application.native_asr_models import (
    NEMOTRON_MODEL_ID,
    PARAKEET_MODEL_ID,
    QWEN_ASR_MODEL_ID,
)

ASR_MODEL_LARGE_V3 = 'large-v3'
ASR_MODEL_LARGE_V3_TURBO = 'large-v3-turbo'
ASR_MODEL_RU_LARGE_V3 = 'bzikst/faster-whisper-large-v3-russian'
ASR_MODEL_RU_PODLODKA_TURBO = 'bzikst/faster-whisper-podlodka-turbo'
ASR_MODEL_MEDIUM = 'medium'
ASR_MODEL_SMALL = 'small'
ASR_MODEL_NEMOTRON_STREAMING = NEMOTRON_MODEL_ID

ASR_MODEL_NAMES = (
    ASR_MODEL_NEMOTRON_STREAMING,
    PARAKEET_MODEL_ID,
    QWEN_ASR_MODEL_ID,
    ASR_MODEL_LARGE_V3,
    ASR_MODEL_LARGE_V3_TURBO,
    ASR_MODEL_RU_LARGE_V3,
    ASR_MODEL_RU_PODLODKA_TURBO,
    ASR_MODEL_MEDIUM,
    ASR_MODEL_SMALL,
)


_ASR_CANDIDATES = {
    'ultra': {
        'ru': (
            ASR_MODEL_NEMOTRON_STREAMING,
            ASR_MODEL_RU_PODLODKA_TURBO,
            ASR_MODEL_SMALL,
        ),
        'default': (
            ASR_MODEL_NEMOTRON_STREAMING,
            ASR_MODEL_LARGE_V3_TURBO,
            ASR_MODEL_SMALL,
        ),
    },
    'fast': {
        'ru': (
            ASR_MODEL_RU_PODLODKA_TURBO,
            ASR_MODEL_LARGE_V3_TURBO,
            ASR_MODEL_SMALL,
            ASR_MODEL_MEDIUM,
        ),
        'default': (
            ASR_MODEL_LARGE_V3_TURBO,
            ASR_MODEL_SMALL,
            ASR_MODEL_MEDIUM,
            ASR_MODEL_RU_PODLODKA_TURBO,
        ),
    },
    'quality': {
        'ru': (
            ASR_MODEL_RU_LARGE_V3,
            ASR_MODEL_LARGE_V3,
            ASR_MODEL_RU_PODLODKA_TURBO,
        ),
        'default': (
            ASR_MODEL_LARGE_V3,
            ASR_MODEL_RU_LARGE_V3,
            ASR_MODEL_LARGE_V3_TURBO,
        ),
    },
    'balanced': {
        'ru': (
            ASR_MODEL_RU_PODLODKA_TURBO,
            ASR_MODEL_MEDIUM,
            ASR_MODEL_LARGE_V3_TURBO,
        ),
        'default': (
            ASR_MODEL_MEDIUM,
            ASR_MODEL_LARGE_V3_TURBO,
            ASR_MODEL_SMALL,
        ),
    },
}

_ACCEPTABLE_CURRENT_MODELS = {
    'ultra': {
        'ru': (
            ASR_MODEL_NEMOTRON_STREAMING,
            ASR_MODEL_RU_PODLODKA_TURBO,
            ASR_MODEL_SMALL,
        ),
        'default': (
            ASR_MODEL_NEMOTRON_STREAMING,
            ASR_MODEL_LARGE_V3_TURBO,
            ASR_MODEL_SMALL,
        ),
    },
    'fast': {
        'ru': (
            ASR_MODEL_RU_PODLODKA_TURBO,
            ASR_MODEL_LARGE_V3_TURBO,
            ASR_MODEL_SMALL,
        ),
        'default': (
            ASR_MODEL_LARGE_V3_TURBO,
            ASR_MODEL_SMALL,
        ),
    },
    'quality': {
        'ru': (
            ASR_MODEL_RU_LARGE_V3,
            ASR_MODEL_LARGE_V3,
        ),
        'default': (
            ASR_MODEL_LARGE_V3,
            ASR_MODEL_RU_LARGE_V3,
        ),
    },
    'balanced': {
        'ru': (
            ASR_MODEL_RU_PODLODKA_TURBO,
            ASR_MODEL_MEDIUM,
            ASR_MODEL_LARGE_V3_TURBO,
        ),
        'default': (
            ASR_MODEL_MEDIUM,
            ASR_MODEL_LARGE_V3_TURBO,
            ASR_MODEL_SMALL,
        ),
    },
}

_POLICY_MODES = {
    PROFILE_ULTRA_FAST.lower(): 'ultra',
    PROFILE_REALTIME.lower(): 'fast',
    PROFILE_QUALITY.lower(): 'quality',
    PROFILE_CUSTOM.lower(): 'custom',
    PROFILE_BALANCED.lower(): 'balanced',
}


@dataclass(frozen=True)
class ModelPolicyDecision:
    asr_model: str
    codex_profile_id: str


def recommend(
    *,
    asr_profile: str,
    language: str,
    current_asr_model: str,
    available_asr_models: Sequence[str] = ASR_MODEL_NAMES,
    codex_profiles: Sequence[CodexProfile] = (),
    current_codex_profile_id: str = '',
) -> ModelPolicyDecision:
    return ModelPolicyDecision(
        asr_model=recommend_asr_model(
            asr_profile=asr_profile,
            language=language,
            current_model=current_asr_model,
            available_models=available_asr_models,
        ),
        codex_profile_id=recommend_codex_profile_id(
            asr_profile=asr_profile,
            profiles=codex_profiles,
            current_profile_id=current_codex_profile_id,
        ),
    )


def recommend_asr_model(
    *,
    asr_profile: str,
    language: str,
    current_model: str,
    available_models: Sequence[str] = ASR_MODEL_NAMES,
) -> str:
    available = dict.fromkeys(
        str(model) for model in available_models if str(model).strip()
    )
    current = str(current_model or '').strip()
    mode = _policy_mode(asr_profile)

    if mode == 'custom':
        return (
            current
            if current in available
            else _first_available(ASR_MODEL_NAMES, available, current)
        )

    compatible = _asr_candidates(mode=mode, language=language)
    if mode == 'ultra':
        return _first_available(compatible, available, current)
    if current in available and current in _acceptable_current_models(
        mode=mode, language=language
    ):
        return current
    return _first_available(compatible, available, current)


def recommend_codex_profile_id(
    *,
    asr_profile: str,
    profiles: Sequence[CodexProfile],
    current_profile_id: str,
) -> str:
    profile_list = [profile for profile in profiles if _profile_id(profile)]
    if not profile_list:
        return ''

    current = str(current_profile_id or '').strip()
    current_exists = any(
        _profile_id(profile) == current for profile in profile_list
    )
    mode = _policy_mode(asr_profile)
    if mode in ('custom', 'balanced') and current_exists:
        return current

    desired_kind = 'quality' if mode == 'quality' else 'fast'
    for profile in profile_list:
        if _codex_profile_kind(profile) == desired_kind:
            return _profile_id(profile)

    return current if current_exists else _profile_id(profile_list[0])


def _asr_candidates(*, mode: str, language: str) -> tuple[str, ...]:
    normalized_language = str(language or '').strip().lower()
    language_key = (
        'ru' if normalized_language in {'ru', 'russian'} else 'default'
    )
    choices = _ASR_CANDIDATES.get(mode, _ASR_CANDIDATES['balanced'])
    return choices[language_key]


def _acceptable_current_models(*, mode: str, language: str) -> tuple[str, ...]:
    normalized_language = str(language or '').strip().lower()
    language_key = (
        'ru' if normalized_language in {'ru', 'russian'} else 'default'
    )
    choices = _ACCEPTABLE_CURRENT_MODELS.get(
        mode, _ACCEPTABLE_CURRENT_MODELS['balanced']
    )
    return choices[language_key]


def _policy_mode(asr_profile: str) -> str:
    profile = str(asr_profile or '').strip().lower()
    return _POLICY_MODES.get(profile, 'balanced')


def _first_available(
    candidates: Sequence[str], available: Mapping[str, None], fallback: str
) -> str:
    for candidate in candidates:
        if candidate in available:
            return candidate
    return fallback if fallback else next(iter(available), '')


def _profile_id(profile: CodexProfile) -> str:
    return str(profile.id or '').strip()


def _codex_profile_kind(profile: CodexProfile) -> str:
    haystack = (
        f'{profile.id} {profile.label} {profile.reasoning_effort}'
    ).lower()
    effort = str(profile.reasoning_effort or '').strip().lower()
    if (
        'deep' in haystack
        or 'quality' in haystack
        or effort in {'high', 'xhigh', 'extra high'}
    ):
        return 'quality'
    if 'fast' in haystack or 'quick' in haystack or effort == 'low':
        return 'fast'
    return 'balanced'


class ModelOrchestrator:
    """Compatibility facade; runtime code uses the module functions."""

    recommend = staticmethod(recommend)
    recommend_asr_model = staticmethod(recommend_asr_model)
    recommend_codex_profile_id = staticmethod(recommend_codex_profile_id)
    _asr_candidates = staticmethod(_asr_candidates)
    _acceptable_current_models = staticmethod(_acceptable_current_models)
    _policy_mode = staticmethod(_policy_mode)
    _first_available = staticmethod(_first_available)
    _profile_id = staticmethod(_profile_id)
    _codex_profile_kind = staticmethod(_codex_profile_kind)
