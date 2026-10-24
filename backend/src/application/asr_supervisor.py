from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from application.asr_session import ASRSessionSettings
from application.model_policy import (
    ASR_MODEL_LARGE_V3_TURBO,
    ASR_MODEL_RU_PODLODKA_TURBO,
    ASR_MODEL_SMALL,
)
from application.supervision import SupervisionReport, supervision_report


@dataclass(frozen=True)
class ASRStartupAttempt:
    label: str
    settings: ASRSessionSettings
    degraded: bool = False


class ASRStartupPolicy(Protocol):
    def build_attempts(
        self, primary: ASRSessionSettings
    ) -> list[ASRStartupAttempt]: ...

    def success_report(
        self, attempt: ASRStartupAttempt, errors: list[str]
    ) -> SupervisionReport: ...

    def failure_report(self, errors: list[str]) -> SupervisionReport: ...


COMPONENT = 'asr'


def build_attempts(primary: ASRSessionSettings) -> list[ASRStartupAttempt]:
    attempts = [
        ASRStartupAttempt(label='primary', settings=primary, degraded=False)
    ]

    fast_model = _fast_model_for_language(primary.language)
    fast_settings = replace(
        primary,
        model_name=fast_model,
        compute_type='int8_float16',
        num_workers=max(2, int(primary.num_workers)),
        beam_size=1,
        max_segment_s=min(float(primary.max_segment_s), 3.0),
        overlap_ms=min(float(primary.overlap_ms), 80.0),
        overload_strategy='drop_old',
        overload_beam_cap=1,
        overload_max_segment_s=min(float(primary.overload_max_segment_s), 2.5),
        overload_overlap_ms=min(float(primary.overload_overlap_ms), 60.0),
    )
    _append_unique(
        attempts,
        ASRStartupAttempt(
            label='fast-fallback', settings=fast_settings, degraded=True
        ),
    )

    cpu_settings = replace(
        primary,
        model_name=ASR_MODEL_SMALL,
        device='cpu',
        compute_type='int8',
        cpu_threads=max(4, int(primary.cpu_threads)),
        num_workers=1,
        beam_size=1,
        max_segment_s=min(float(primary.max_segment_s), 4.0),
        overlap_ms=min(float(primary.overlap_ms), 80.0),
        overload_strategy='drop_old',
        overload_beam_cap=1,
        overload_max_segment_s=min(float(primary.overload_max_segment_s), 3.0),
        overload_overlap_ms=min(float(primary.overload_overlap_ms), 60.0),
    )
    _append_unique(
        attempts,
        ASRStartupAttempt(
            label='cpu-small-fallback', settings=cpu_settings, degraded=True
        ),
    )
    return attempts


def success_report(
    attempt: ASRStartupAttempt, errors: list[str]
) -> SupervisionReport:
    return supervision_report(
        component=COMPONENT,
        active_attempt=attempt.label,
        fallback_used=bool(attempt.degraded),
        errors=errors,
    )


def failure_report(errors: list[str]) -> SupervisionReport:
    return supervision_report(
        component=COMPONENT,
        active_attempt='none',
        fallback_used=True,
        errors=errors,
        failed=True,
    )


def _fast_model_for_language(language: str) -> str:
    lang = str(language or '').strip().lower()
    if lang in {'ru', 'russian'}:
        return ASR_MODEL_RU_PODLODKA_TURBO
    return ASR_MODEL_LARGE_V3_TURBO


def _append_unique(
    attempts: list[ASRStartupAttempt], attempt: ASRStartupAttempt
) -> None:
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


class ASRStartupSupervisor:
    """Compatibility facade; runtime code uses the module functions."""

    component = COMPONENT
    build_attempts = staticmethod(build_attempts)
    success_report = staticmethod(success_report)
    failure_report = staticmethod(failure_report)
    _fast_model_for_language = staticmethod(_fast_model_for_language)
    _append_unique = staticmethod(_append_unique)
