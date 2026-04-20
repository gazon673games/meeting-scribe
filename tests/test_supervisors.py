from __future__ import annotations

import unittest

from application.asr_session import ASRSessionSettings
from application.asr_supervisor import ASRStartupSupervisor
from application.assistant_supervisor import AssistantFallbackSupervisor
from application.model_policy import ASR_MODEL_RU_PODLODKA_TURBO, ASR_MODEL_SMALL
from application.supervision import SupervisionStatus


def _settings(**overrides) -> ASRSessionSettings:
    values = {
        "language": "ru",
        "mode": "split",
        "model_name": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
        "beam_size": 6,
        "endpoint_silence_ms": 900.0,
        "max_segment_s": 12.0,
        "overlap_ms": 320.0,
        "vad_energy_threshold": 0.0052,
        "overload_strategy": "keep_all",
        "overload_enter_qsize": 22,
        "overload_exit_qsize": 8,
        "overload_hard_qsize": 40,
        "overload_beam_cap": 3,
        "overload_max_segment_s": 6.0,
        "overload_overlap_ms": 160.0,
        "asr_language": "ru",
        "asr_initial_prompt": None,
    }
    values.update(overrides)
    return ASRSessionSettings(**values)


class SupervisorTests(unittest.TestCase):
    def test_asr_startup_supervisor_adds_fast_and_cpu_fallbacks(self) -> None:
        attempts = ASRStartupSupervisor().build_attempts(_settings())

        self.assertEqual(attempts[0].label, "primary")
        self.assertFalse(attempts[0].degraded)
        self.assertEqual(attempts[1].settings.model_name, ASR_MODEL_RU_PODLODKA_TURBO)
        self.assertEqual(attempts[1].settings.compute_type, "int8_float16")
        self.assertEqual(attempts[1].settings.beam_size, 2)
        self.assertTrue(attempts[1].degraded)
        self.assertEqual(attempts[-1].settings.model_name, ASR_MODEL_SMALL)
        self.assertEqual(attempts[-1].settings.device, "cpu")
        self.assertEqual(attempts[-1].settings.compute_type, "int8")

    def test_assistant_supervisor_builds_primary_and_fallback_attempts(self) -> None:
        attempts = AssistantFallbackSupervisor().build_attempts(
            max_log_chars=4000,
            timeout_s=35,
            fallback_max_log_chars=2000,
            fallback_timeout_s=20,
        )

        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0].label, "primary")
        self.assertFalse(attempts[0].fallback)
        self.assertEqual(attempts[0].max_log_chars, 4000)
        self.assertEqual(attempts[1].label, "fallback")
        self.assertTrue(attempts[1].fallback)
        self.assertEqual(attempts[1].max_log_chars, 2000)
        self.assertEqual(attempts[1].timeout_s, 20)

    def test_supervisors_report_degraded_and_failed_status(self) -> None:
        asr_supervisor = ASRStartupSupervisor()
        asr_attempts = asr_supervisor.build_attempts(_settings())

        degraded = asr_supervisor.success_report(asr_attempts[1], ["primary failed"])
        failed = asr_supervisor.failure_report(["primary failed", "fallback failed"])

        self.assertEqual(degraded.status, SupervisionStatus.DEGRADED)
        self.assertTrue(degraded.degraded)
        self.assertEqual(degraded.active_attempt, "fast-fallback")
        self.assertEqual(failed.status, SupervisionStatus.FAILED)
        self.assertTrue(failed.failed)

        assistant_supervisor = AssistantFallbackSupervisor()
        assistant_attempts = assistant_supervisor.build_attempts(
            max_log_chars=4000,
            timeout_s=35,
            fallback_max_log_chars=2000,
            fallback_timeout_s=20,
        )
        assistant_report = assistant_supervisor.success_report(assistant_attempts[1], ["timeout"])

        self.assertEqual(assistant_report.status, SupervisionStatus.DEGRADED)
        self.assertEqual(assistant_report.component, "assistant")


if __name__ == "__main__":
    unittest.main()
