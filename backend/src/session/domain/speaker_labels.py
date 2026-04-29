from __future__ import annotations


ME_SPEAKER_LABEL = "Me"
REMOTE_SPEAKER_LABEL = "Remote"


def default_speaker_label_for_source_kind(kind: str) -> str:
    normalized = str(kind or "").strip().lower()
    if normalized in {"input", "mic", "microphone"}:
        return ME_SPEAKER_LABEL
    if normalized in {"loopback", "system", "desktop", "desktop_audio", "process", "app", "application"}:
        return REMOTE_SPEAKER_LABEL
    return ""
