from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from application.assistant_supervisor import AssistantFallbackSupervisor
from application.assistant_use_case import AssistantRequestUseCase
from assistant.application.provider import (
    AssistantExecutionSettings,
    AssistantProviderError,
    AssistantProviderInfo,
    AssistantProviderLoginResult,
    AssistantProviderPingResult,
    AssistantProviderResult,
)
from assistant.application.service import AssistantApplicationService, AssistantRuntimeOptions
from assistant.domain.aggregate import AssistantJobAggregate


class _Provider:
    provider_id = "mock"
    provider_label = "Mock"

    def status(self, settings):  # noqa: ANN001
        return AssistantProviderInfo(id=self.provider_id, label=self.provider_label, available=True, models=["m"])

    def start_login(self, settings, *, device_auth=False):  # noqa: ANN001
        return AssistantProviderLoginResult(
            id=self.provider_id,
            label=self.provider_label,
            started=bool(device_auth),
            local_home="home",
        )

    def ping(self, settings):  # noqa: ANN001
        return AssistantProviderPingResult(id=self.provider_id, label=self.provider_label, ok=True, status_code=204)


class _NoOptionalProvider:
    provider_id = "plain"
    provider_label = "Plain"

    def run(self, request):  # noqa: ANN001
        return AssistantProviderResult(ok=True, profile="p", cmd=request.original_cmd, text="ok", dt_s=0)


class _RaisingProvider(_Provider):
    provider_id = "raising"
    provider_label = "Raising"

    def status(self, settings):  # noqa: ANN001
        raise RuntimeError("status failed")

    def start_login(self, settings, *, device_auth=False):  # noqa: ANN001
        raise RuntimeError("login failed")

    def ping(self, settings):  # noqa: ANN001
        raise RuntimeError("ping failed")


class AssistantProviderFacadeTests(unittest.TestCase):
    def test_provider_result_info_login_and_ping_objects_serialize_errors(self) -> None:
        self.assertTrue(AssistantProviderInfo("id", "Label", True, models=["a"]).as_dict()["available"])
        self.assertTrue(AssistantProviderLoginResult("id", "Label", True).as_dict()["started"])
        self.assertEqual(AssistantProviderPingResult("id", "Label", False, status_code=503).as_dict()["statusCode"], 503)

        result = AssistantProviderResult(
            ok=False,
            profile="p",
            cmd="cmd",
            text="failed",
            dt_s=0,
            error_code="boom",
            retryable=True,
            suggestion="retry",
            details="details",
        )
        self.assertEqual(result.error, AssistantProviderError("boom", "failed", True, "retry", "details"))

    def test_assistant_use_case_reports_status_login_and_ping_edges(self) -> None:
        settings = AssistantExecutionSettings(command_tokens=["codex"], path_hints=[], proxy="", timeout_s=1)
        use_case = AssistantRequestUseCase([_Provider(), _NoOptionalProvider(), _RaisingProvider()], context_reader=None)

        statuses = use_case.provider_statuses(settings)
        self.assertEqual([status.id for status in statuses], ["mock", "plain", "raising"])
        self.assertEqual(statuses[2].error_code, "provider_status_error")
        self.assertEqual(use_case.start_provider_login("missing", settings).error_code, "provider_unavailable")
        self.assertEqual(use_case.start_provider_login("plain", settings).error_code, "login_not_supported")
        self.assertTrue(use_case.start_provider_login("mock", settings, device_auth=True).started)
        self.assertEqual(use_case.start_provider_login("raising", settings).error_code, "login_start_failed")
        self.assertEqual(use_case.ping_provider("missing", settings).error_code, "provider_unavailable")
        self.assertEqual(use_case.ping_provider("plain", settings).error_code, "ping_not_supported")
        self.assertTrue(use_case.ping_provider("mock", settings).ok)
        self.assertEqual(use_case.ping_provider("raising", settings).error_code, "ping_failed")

    def test_application_service_delegates_provider_facades_with_runtime_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            use_case = AssistantRequestUseCase(_Provider(), context_reader=None)
            service = AssistantApplicationService(use_case)
            options = AssistantRuntimeOptions(
                project_root=Path(tmp),
                default_max_log_chars=2000,
                answer_keyword="ANSWER",
                command_tokens=["codex"],
                path_hints=["hint"],
                proxy="",
                default_timeout_s=30,
                profiles=["profile"],
            )

            self.assertEqual(service.provider_statuses(options=options)[0].id, "mock")
            self.assertTrue(service.start_provider_login("mock", options=options, device_auth=True).started)
            self.assertTrue(service.ping_provider("mock", options=options).ok)

        report = AssistantFallbackSupervisor().failure_report(["bad"])
        self.assertTrue(report.failed)
        self.assertEqual(AssistantJobAggregate().state.value, "idle")


if __name__ == "__main__":
    unittest.main()
