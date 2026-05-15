from __future__ import annotations

import unittest
from pathlib import Path

from application.asr_session import ASRSessionSettings
from application.codex_assistant import CodexAssistantResult
from application.codex_config import CodexProfile
from application.commands import InvokeAssistantCommand
from application.event_types import CodexFallbackStartedEvent, CodexResultEvent
from assistant.application.service import AssistantApplicationService, AssistantRuntimeOptions
from transcription.application.startup_service import TranscriptionStartupService


class _AssistantUseCase:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request) -> CodexAssistantResult:
        self.calls += 1
        if self.calls == 1:
            return CodexAssistantResult(ok=False, profile=request.profile.label, cmd=request.user_text, text="timeout", dt_s=1.0)
        return CodexAssistantResult(ok=True, profile=request.profile.label, cmd=request.user_text, text="ok", dt_s=2.0)


class _Runtime:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


class _RuntimeFactory:
    def __init__(self) -> None:
        self.calls = 0

    def build(self, settings, *, tap_queue, project_root: Path, event_queue=None) -> _Runtime:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("cuda failed")
        return _Runtime()


def _asr_settings() -> ASRSessionSettings:
    return ASRSessionSettings(
        language="ru",
        mode="split",
        model_name="large-v3",
        device="cuda",
        compute_type="float16",
        cpu_threads=0,
        num_workers=1,
        beam_size=6,
        endpoint_silence_ms=900.0,
        max_segment_s=12.0,
        overlap_ms=320.0,
        vad_energy_threshold=0.0052,
        overload_strategy="keep_all",
        overload_enter_qsize=22,
        overload_exit_qsize=8,
        overload_hard_qsize=40,
        overload_beam_cap=3,
        overload_max_segment_s=6.0,
        overload_overlap_ms=160.0,
        asr_language="ru",
        asr_initial_prompt=None,
    )


class ApplicationServiceTests(unittest.TestCase):
    def test_assistant_service_publishes_fallback_and_result(self) -> None:
        service = AssistantApplicationService(_AssistantUseCase())  # type: ignore[arg-type]
        profile = CodexProfile(id="fast", label="Fast", prompt="")
        command = InvokeAssistantCommand(
            profile=profile,
            request_text="ANSWER",
            source_label="answer",
            context_source="transcript",
            context_label="current transcript",
            context_text="question",
            max_log_chars=4000,
            timeout_s=35,
            fallback_max_log_chars=2000,
            fallback_timeout_s=20,
        )
        events: list[object] = []

        service.execute(
            command,
            options=AssistantRuntimeOptions(
                project_root=Path("."),
                default_max_log_chars=24000,
                answer_keyword="ANSWER",
                command_tokens=["codex"],
                path_hints=[],
                proxy="",
                default_timeout_s=90,
            ),
            publish_event=events.append,
        )

        self.assertIsInstance(events[0], CodexFallbackStartedEvent)
        self.assertIsInstance(events[1], CodexResultEvent)
        self.assertTrue(events[1].ok)  # type: ignore[attr-defined]
        self.assertTrue(service.last_supervision_report.degraded)  # type: ignore[union-attr]

    def test_transcription_startup_service_returns_degraded_fallback_result(self) -> None:
        factory = _RuntimeFactory()

        result = TranscriptionStartupService().start(
            _asr_settings(),
            runtime_factory=factory,
            tap_queue=None,
            project_root=Path("."),
            event_queue=None,
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.degraded)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(factory.calls, 2)


if __name__ == "__main__":
    unittest.main()
