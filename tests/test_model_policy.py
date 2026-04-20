from __future__ import annotations

import unittest

from application.asr_profiles import PROFILE_BALANCED, PROFILE_CUSTOM, PROFILE_QUALITY, PROFILE_REALTIME
from application.codex_config import CodexProfile
from application.model_policy import (
    ASR_MODEL_LARGE_V3,
    ASR_MODEL_LARGE_V3_TURBO,
    ASR_MODEL_MEDIUM,
    ASR_MODEL_RU_LARGE_V3,
    ASR_MODEL_RU_PODLODKA_TURBO,
    ASR_MODEL_SMALL,
    ModelOrchestrator,
)


class ModelOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = ModelOrchestrator()

    def test_realtime_ru_prefers_russian_turbo(self) -> None:
        model = self.orchestrator.recommend_asr_model(
            asr_profile=PROFILE_REALTIME,
            language="ru",
            current_model=ASR_MODEL_LARGE_V3,
        )

        self.assertEqual(model, ASR_MODEL_RU_PODLODKA_TURBO)

    def test_realtime_en_prefers_generic_turbo(self) -> None:
        model = self.orchestrator.recommend_asr_model(
            asr_profile=PROFILE_REALTIME,
            language="en",
            current_model=ASR_MODEL_MEDIUM,
        )

        self.assertEqual(model, ASR_MODEL_LARGE_V3_TURBO)

    def test_quality_ru_prefers_russian_large(self) -> None:
        model = self.orchestrator.recommend_asr_model(
            asr_profile=PROFILE_QUALITY,
            language="ru",
            current_model=ASR_MODEL_SMALL,
        )

        self.assertEqual(model, ASR_MODEL_RU_LARGE_V3)

    def test_balanced_keeps_compatible_current_model(self) -> None:
        model = self.orchestrator.recommend_asr_model(
            asr_profile=PROFILE_BALANCED,
            language="ru",
            current_model=ASR_MODEL_MEDIUM,
        )

        self.assertEqual(model, ASR_MODEL_MEDIUM)

    def test_custom_keeps_current_model(self) -> None:
        model = self.orchestrator.recommend_asr_model(
            asr_profile=PROFILE_CUSTOM,
            language="ru",
            current_model=ASR_MODEL_SMALL,
        )

        self.assertEqual(model, ASR_MODEL_SMALL)

    def test_quality_codex_selects_deep_profile(self) -> None:
        profile_id = self.orchestrator.recommend_codex_profile_id(
            asr_profile=PROFILE_QUALITY,
            profiles=[
                CodexProfile(id="interview_fast", label="Interview Fast", prompt="", reasoning_effort="low"),
                CodexProfile(id="interview_deep", label="Interview Deep", prompt="", reasoning_effort="high"),
            ],
            current_profile_id="interview_fast",
        )

        self.assertEqual(profile_id, "interview_deep")

    def test_realtime_codex_selects_fast_profile(self) -> None:
        profile_id = self.orchestrator.recommend_codex_profile_id(
            asr_profile=PROFILE_REALTIME,
            profiles=[
                CodexProfile(id="interview_deep", label="Interview Deep", prompt="", reasoning_effort="high"),
                CodexProfile(id="interview_fast", label="Interview Fast", prompt="", reasoning_effort="low"),
            ],
            current_profile_id="interview_deep",
        )

        self.assertEqual(profile_id, "interview_fast")


if __name__ == "__main__":
    unittest.main()
