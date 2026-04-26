from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from application.asr_profiles import PROFILE_BALANCED, PROFILE_CUSTOM, PROFILE_QUALITY, PROFILE_REALTIME

ASR_MODEL_LARGE_V3 = "large-v3"
ASR_MODEL_LARGE_V3_TURBO = "large-v3-turbo"
ASR_MODEL_RU_LARGE_V3 = "bzikst/faster-whisper-large-v3-russian"
ASR_MODEL_RU_PODLODKA_TURBO = "bzikst/faster-whisper-podlodka-turbo"
ASR_MODEL_MEDIUM = "medium"
ASR_MODEL_SMALL = "small"

ASR_MODEL_NAMES = (
    ASR_MODEL_LARGE_V3,
    ASR_MODEL_LARGE_V3_TURBO,
    ASR_MODEL_RU_LARGE_V3,
    ASR_MODEL_RU_PODLODKA_TURBO,
    ASR_MODEL_MEDIUM,
    ASR_MODEL_SMALL,
)


@dataclass(frozen=True)
class ModelPolicyDecision:
    asr_model: str
    codex_profile_id: str


class ModelOrchestrator:
    def recommend(
        self,
        *,
        asr_profile: str,
        language: str,
        current_asr_model: str,
        available_asr_models: Sequence[str] = ASR_MODEL_NAMES,
        codex_profiles: Sequence[Any] = (),
        current_codex_profile_id: str = "",
    ) -> ModelPolicyDecision:
        return ModelPolicyDecision(
            asr_model=self.recommend_asr_model(
                asr_profile=asr_profile,
                language=language,
                current_model=current_asr_model,
                available_models=available_asr_models,
            ),
            codex_profile_id=self.recommend_codex_profile_id(
                asr_profile=asr_profile,
                profiles=codex_profiles,
                current_profile_id=current_codex_profile_id,
            ),
        )

    def recommend_asr_model(
        self,
        *,
        asr_profile: str,
        language: str,
        current_model: str,
        available_models: Sequence[str] = ASR_MODEL_NAMES,
    ) -> str:
        available = {str(model) for model in available_models if str(model).strip()}
        current = str(current_model or "").strip()
        mode = self._policy_mode(asr_profile)

        if mode == "custom":
            return current if current in available else self._first_available(ASR_MODEL_NAMES, available, current)

        compatible = self._asr_candidates(mode=mode, language=language)
        if current in available and current in self._acceptable_current_models(mode=mode, language=language):
            return current
        return self._first_available(compatible, available, current)

    def recommend_codex_profile_id(
        self,
        *,
        asr_profile: str,
        profiles: Sequence[Any],
        current_profile_id: str,
    ) -> str:
        profile_list = [profile for profile in profiles if self._profile_id(profile)]
        if not profile_list:
            return ""

        current = str(current_profile_id or "").strip()
        current_exists = any(self._profile_id(profile) == current for profile in profile_list)
        mode = self._policy_mode(asr_profile)
        if mode in ("custom", "balanced") and current_exists:
            return current

        desired_kind = "quality" if mode == "quality" else "fast"
        for profile in profile_list:
            if self._codex_profile_kind(profile) == desired_kind:
                return self._profile_id(profile)

        return current if current_exists else self._profile_id(profile_list[0])

    def _asr_candidates(self, *, mode: str, language: str) -> tuple[str, ...]:
        lang = str(language or "").strip().lower()
        ru_first = lang in {"ru", "russian"}

        if mode == "fast":
            if ru_first:
                return (ASR_MODEL_RU_PODLODKA_TURBO, ASR_MODEL_LARGE_V3_TURBO, ASR_MODEL_SMALL, ASR_MODEL_MEDIUM)
            return (ASR_MODEL_LARGE_V3_TURBO, ASR_MODEL_SMALL, ASR_MODEL_MEDIUM, ASR_MODEL_RU_PODLODKA_TURBO)

        if mode == "quality":
            if ru_first:
                return (ASR_MODEL_RU_LARGE_V3, ASR_MODEL_LARGE_V3, ASR_MODEL_RU_PODLODKA_TURBO)
            return (ASR_MODEL_LARGE_V3, ASR_MODEL_RU_LARGE_V3, ASR_MODEL_LARGE_V3_TURBO)

        if ru_first:
            return (ASR_MODEL_RU_PODLODKA_TURBO, ASR_MODEL_MEDIUM, ASR_MODEL_LARGE_V3_TURBO)
        return (ASR_MODEL_MEDIUM, ASR_MODEL_LARGE_V3_TURBO, ASR_MODEL_SMALL)

    def _acceptable_current_models(self, *, mode: str, language: str) -> tuple[str, ...]:
        lang = str(language or "").strip().lower()
        ru_first = lang in {"ru", "russian"}

        if mode == "fast":
            if ru_first:
                return (ASR_MODEL_RU_PODLODKA_TURBO, ASR_MODEL_LARGE_V3_TURBO, ASR_MODEL_SMALL)
            return (ASR_MODEL_LARGE_V3_TURBO, ASR_MODEL_SMALL)

        if mode == "quality":
            if ru_first:
                return (ASR_MODEL_RU_LARGE_V3, ASR_MODEL_LARGE_V3)
            return (ASR_MODEL_LARGE_V3, ASR_MODEL_RU_LARGE_V3)

        if ru_first:
            return (ASR_MODEL_RU_PODLODKA_TURBO, ASR_MODEL_MEDIUM, ASR_MODEL_LARGE_V3_TURBO)
        return (ASR_MODEL_MEDIUM, ASR_MODEL_LARGE_V3_TURBO, ASR_MODEL_SMALL)

    def _policy_mode(self, asr_profile: str) -> str:
        profile = str(asr_profile or "").strip().lower()
        if profile == PROFILE_REALTIME.lower():
            return "fast"
        if profile == PROFILE_QUALITY.lower():
            return "quality"
        if profile == PROFILE_CUSTOM.lower():
            return "custom"
        if profile == PROFILE_BALANCED.lower():
            return "balanced"
        return "balanced"

    @staticmethod
    def _first_available(candidates: Sequence[str], available: set[str], fallback: str) -> str:
        for candidate in candidates:
            if candidate in available:
                return candidate
        return fallback if fallback else next(iter(available), "")

    @staticmethod
    def _profile_id(profile: Any) -> str:
        return str(getattr(profile, "id", "") or "").strip()

    @staticmethod
    def _codex_profile_kind(profile: Any) -> str:
        haystack = (
            f"{getattr(profile, 'id', '')} "
            f"{getattr(profile, 'label', '')} "
            f"{getattr(profile, 'reasoning_effort', '')}"
        ).lower()
        effort = str(getattr(profile, "reasoning_effort", "") or "").strip().lower()
        if "deep" in haystack or "quality" in haystack or effort in {"high", "xhigh", "extra high"}:
            return "quality"
        if "fast" in haystack or "quick" in haystack or effort == "low":
            return "fast"
        return "balanced"
