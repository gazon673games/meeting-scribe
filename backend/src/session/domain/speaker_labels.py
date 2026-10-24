from __future__ import annotations

ME_SPEAKER_LABEL = 'Me'
REMOTE_SPEAKER_LABEL = 'Remote'


_SOURCE_SPEAKER_LABELS = {
    'input': ME_SPEAKER_LABEL,
    'mic': ME_SPEAKER_LABEL,
    'microphone': ME_SPEAKER_LABEL,
    'loopback': REMOTE_SPEAKER_LABEL,
    'system': REMOTE_SPEAKER_LABEL,
    'desktop': REMOTE_SPEAKER_LABEL,
    'desktop_audio': REMOTE_SPEAKER_LABEL,
    'process': REMOTE_SPEAKER_LABEL,
    'app': REMOTE_SPEAKER_LABEL,
    'application': REMOTE_SPEAKER_LABEL,
}


def default_speaker_label_for_source_kind(kind: str) -> str:
    normalized = str(kind or '').strip().lower()
    return _SOURCE_SPEAKER_LABELS.get(normalized, '')
