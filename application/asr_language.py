from __future__ import annotations

from typing import Optional

SUPPORTED_ASR_LANGUAGES = ("ru", "en", "auto")


def normalize_asr_language(raw: str) -> str:
    value = (raw or "ru").strip().lower()
    if value not in SUPPORTED_ASR_LANGUAGES:
        return "ru"
    return value


def runtime_asr_language(ui_language: str) -> Optional[str]:
    value = normalize_asr_language(ui_language)
    return None if value == "auto" else value


def initial_prompt_for_language(ui_language: str) -> Optional[str]:
    value = normalize_asr_language(ui_language)
    if value == "ru":
        return (
            "\u0422\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u0431\u0438\u0440\u0443\u0439 "
            "\u0440\u0430\u0437\u0433\u043e\u0432\u043e\u0440\u043d\u0443\u044e "
            "\u0440\u0443\u0441\u0441\u043a\u0443\u044e \u0440\u0435\u0447\u044c. "
            "\u0421\u043e\u0445\u0440\u0430\u043d\u044f\u0439 \u0447\u0438\u0441\u043b\u0430, "
            "\u0438\u043c\u0435\u043d\u0430, \u0442\u0435\u0440\u043c\u0438\u043d\u044b. "
            "\u0421\u0442\u0430\u0432\u044c \u043f\u0443\u043d\u043a\u0442\u0443\u0430\u0446\u0438\u044e."
        )
    if value == "en":
        return "Transcribe conversational English. Keep numbers, names, and technical terms. Add punctuation."
    return None
