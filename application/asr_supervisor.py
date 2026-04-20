from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List

from application.asr_session import ASRSessionSettings
from application.model_policy import (
    ASR_MODEL_LARGE_V3_TURBO,
    ASR_MODEL_RU_PODLODKA_TURBO,
    ASR_MODEL_SMALL,
)


@dataclass(frozen=True)
class ASRStartupAttempt:
    label: str
    settings: ASRSessionSettings
    degraded: bool = False


class ASRStartupSupervisor:
    def build_attempts(self, primary: ASRSessionSettings) -> List[ASRStartupAttempt]:
        attempts = [ASRStartupAttempt(label="primary", settings=primary, degraded=False)]

        fast_model = self._fast_model_for_language(primary.language)
        fast_settings = replace(
            primary,
            model_name=fast_model,
            compute_type="int8_float16",
            beam_size=min(int(primary.beam_size), 2),
            max_segment_s=min(float(primary.max_segment_s), 5.0),
            overlap_ms=min(float(primary.overlap_ms), 120.0),
            overload_strategy="drop_old",
            overload_beam_cap=1,
            overload_max_segment_s=min(float(primary.overload_max_segment_s), 3.5),
            overload_overlap_ms=min(float(primary.overload_overlap_ms), 80.0),
        )
        self._append_unique(attempts, ASRStartupAttempt(label="fast-fallback", settings=fast_settings, degraded=True))

        cpu_settings = replace(
            primary,
            model_name=ASR_MODEL_SMALL,
            device="cpu",
            compute_type="int8",
            beam_size=1,
            max_segment_s=min(float(primary.max_segment_s), 4.0),
            overlap_ms=min(float(primary.overlap_ms), 80.0),
            overload_strategy="drop_old",
            overload_beam_cap=1,
            overload_max_segment_s=min(float(primary.overload_max_segment_s), 3.0),
            overload_overlap_ms=min(float(primary.overload_overlap_ms), 60.0),
        )
        self._append_unique(attempts, ASRStartupAttempt(label="cpu-small-fallback", settings=cpu_settings, degraded=True))
        return attempts

    @staticmethod
    def _fast_model_for_language(language: str) -> str:
        lang = str(language or "").strip().lower()
        if lang in {"ru", "russian"}:
            return ASR_MODEL_RU_PODLODKA_TURBO
        return ASR_MODEL_LARGE_V3_TURBO

    @staticmethod
    def _append_unique(attempts: List[ASRStartupAttempt], attempt: ASRStartupAttempt) -> None:
        key = (
            attempt.settings.model_name,
            attempt.settings.device,
            attempt.settings.compute_type,
            int(attempt.settings.beam_size),
        )
        for existing in attempts:
            existing_key = (
                existing.settings.model_name,
                existing.settings.device,
                existing.settings.compute_type,
                int(existing.settings.beam_size),
            )
            if existing_key == key:
                return
        attempts.append(attempt)
