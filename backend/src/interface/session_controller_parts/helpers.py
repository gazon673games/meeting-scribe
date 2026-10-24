from __future__ import annotations

import time
from typing import Any

from application.recording import WavRecorder
from shared.domain.mappings import first_param as first_param

_SOURCE_KIND_ALIASES = {
    'loopback': 'loopback',
    'system': 'loopback',
    'desktop': 'loopback',
    'desktop_audio': 'loopback',
    'input': 'input',
    'mic': 'input',
    'microphone': 'input',
    'process': 'process',
    'app': 'process',
    'application': 'process',
    'per_process': 'process',
}

_DEFAULT_SOURCE_NAMES = {
    'loopback': 'desktop_audio',
    'process': 'app_audio',
}

_SPEAKER_IDENTITY_ALIASES = {
    'enabled': ('speakerIdentityEnabled', 'speaker_identity_enabled'),
    'persistent_profiles_enabled': (
        'speakerIdentityPersistentProfilesEnabled',
        'speaker_identity_persistent_profiles_enabled',
    ),
    'backend': ('speakerIdentityBackend', 'speaker_identity_backend'),
    'app_data_dir': (
        'speakerIdentityAppDataDir',
        'speaker_identity_app_data_dir',
    ),
    'auto_match_threshold': (
        'speakerIdentityAutoMatchThreshold',
        'speaker_identity_auto_match_threshold',
    ),
    'uncertain_match_threshold': (
        'speakerIdentityUncertainMatchThreshold',
        'speaker_identity_uncertain_match_threshold',
    ),
    'min_speech_ms_for_embedding': (
        'speakerIdentityMinSpeechMsForEmbedding',
        'speaker_identity_min_speech_ms_for_embedding',
    ),
    'min_quality_score': (
        'speakerIdentityMinQualityScore',
        'speaker_identity_min_quality_score',
    ),
}


def normalize_source_kind(kind: str) -> str:
    name = str(kind or '').strip().lower()
    normalized = _SOURCE_KIND_ALIASES.get(name)
    if normalized is None:
        raise ValueError(f'Unsupported source kind: {kind}')
    return normalized


def default_source_name(kind: str) -> str:
    return _DEFAULT_SOURCE_NAMES.get(kind, 'mic')


def safe_float(raw: object, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def optional_float(raw: object) -> float | None:
    try:
        if raw is None:
            return None
        return float(raw)
    except (TypeError, ValueError, OverflowError):
        return None


def rms_to_pct(rms: float) -> int:
    value = max(0.0, min(1.0, float(rms)))
    return max(0, min(100, int((value**0.5) * 100.0)))


def master_record(meters: dict[str, Any]) -> dict[str, Any]:
    master = meters.get('master', {}) if isinstance(meters, dict) else {}
    rms = safe_float(master.get('rms', 0.0), 0.0)
    last_ts = safe_float(master.get('last_ts', 0.0), 0.0)
    return {
        'rms': rms,
        'level': rms_to_pct(rms),
        'active': bool(time.monotonic() - last_ts < 0.6 and rms > 1e-4),
        'lastTs': last_ts,
    }


def drops_record(meters: dict[str, Any], writer: WavRecorder) -> dict[str, Any]:
    drops = meters.get('drops', {}) if isinstance(meters, dict) else {}
    drained = getattr(writer, 'drained_blocks', lambda: 0)
    written = getattr(writer, 'written_blocks', lambda: 0)
    return {
        'droppedOutBlocks': int(
            safe_float(drops.get('dropped_out_blocks', 0), 0.0)
        ),
        'droppedTapBlocks': int(
            safe_float(drops.get('dropped_tap_blocks', 0), 0.0)
        ),
        'drainedBlocks': int(drained()),
        'writtenBlocks': int(written()),
    }


def wav_requested(params: dict[str, Any]) -> bool:
    return bool(first_param(params, 'wavEnabled', 'wav_enabled', default=False))


def join_words(words: object) -> str:
    if not isinstance(words, list):
        return ''
    return ' '.join(str(w.get('text', '')) for w in words).strip()


def clean_speaker(raw: object) -> str:
    text = str(raw or '').strip()
    if not text or text == 'S?':
        return ''
    return text


def line_speaker_or_stream(line: dict[str, Any]) -> str:
    return clean_speaker(line.get('speaker', '')) or str(
        line.get('stream') or 'mix'
    )


def normalize_diar_backend(raw: object):
    value = str(raw or 'online').strip().lower()
    if value in {'pyannote', 'online', 'nemo', 'sherpa_onnx'}:
        return value
    return 'online'


def safe_bool(raw: object, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return bool(default)
    value = str(raw).strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off', ''}:
        return False
    return bool(default)


def safe_int(raw: object, default: int, lo: int, hi: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        value = int(default)
    return max(int(lo), min(int(hi), value))


def safe_float_clamped(
    raw: object, default: float, lo: float, hi: float
) -> float:
    value = safe_float(raw, default)
    return max(float(lo), min(float(hi), value))


def speaker_identity_params(params: dict[str, Any]) -> dict[str, Any]:
    nested = first_param(
        params, 'speakerIdentity', 'speaker_identity', default={}
    )
    out = dict(nested) if isinstance(nested, dict) else {}
    for key, names in _SPEAKER_IDENTITY_ALIASES.items():
        for name in names:
            if name in params:
                out[key] = params[name]
                break
    return out
