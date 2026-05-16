from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from assistant.application.provider import (
    AssistantProviderInfo,
    AssistantProviderLoginResult,
    AssistantProviderPingResult,
)
from application.codex_config import CodexProfile
from interface.assistant_controller import AssistantController
from interface.assistant_controller_parts.provider_cache import provider_for_profile
from settings.infrastructure.json_config_repository import JsonConfigRepository


class _ImmediateThread:
    def __init__(self, *, target, args=(), kwargs=None, **unused):  # noqa: ANN001
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self) -> None:
        self.target(*self.args, **self.kwargs)


class _ProviderFacadeService:
    def __init__(self, *, raise_status: bool = False) -> None:
        self.raise_status = raise_status

    def provider_statuses(self, *, options):  # noqa: ANN001
        if self.raise_status:
            raise RuntimeError("status failed")
        return [AssistantProviderInfo(id="codex", label="Codex", available=True, login_supported=True)]

    def start_provider_login(self, provider_id: str, *, options, device_auth=False):  # noqa: ANN001
        return AssistantProviderLoginResult(id=provider_id, label=provider_id, started=bool(device_auth))

    def ping_provider(self, provider_id: str, *, options, profile=None):  # noqa: ANN001
        return AssistantProviderPingResult(id=provider_id, label=provider_id, ok=True, status_code=200)


class AssistantControllerFacadeTests(unittest.TestCase):
    def _controller(self, root: Path, service: _ProviderFacadeService) -> AssistantController:
        repository = JsonConfigRepository(root / "config.json")
        repository.write(
            {
                "codex": {
                    "enabled": True,
                    "selected_profile": "fast",
                    "profiles": [{"id": "fast", "label": "Fast", "prompt": "help", "provider": "codex"}],
                }
            }
        )
        return AssistantController(project_root=root, config_repository=repository, assistant_service=service)  # type: ignore[arg-type]

    def test_provider_for_profile_matches_requested_provider_id(self) -> None:
        providers = [{"id": "codex"}, {"id": "ollama"}]

        self.assertIsNone(provider_for_profile([], CodexProfile("p", "Profile", "")))
        self.assertEqual(provider_for_profile(providers, CodexProfile("p", "Profile", "", provider_id="ollama"))["id"], "ollama")
        self.assertEqual(provider_for_profile(providers, None)["id"], "codex")
        self.assertIsNone(provider_for_profile(providers, CodexProfile("p", "Profile", "", provider_id="missing")))

    def test_provider_login_ping_cache_and_event_sink_facades(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            events: list[tuple[str, dict]] = []
            controller = self._controller(root, _ProviderFacadeService())
            controller.set_event_sink(lambda event_type, payload: events.append((event_type, payload)))
            settings = controller._settings()

            records = controller._refresh_provider_records(settings)
            cached = controller._provider_records(settings)
            login = controller.start_provider_login({"providerId": "codex", "deviceAuth": True})
            alias_login = controller.start_login({"providerId": "codex", "deviceAuth": False})
            with patch("interface.assistant_controller.threading.Thread", _ImmediateThread):
                pending = controller.ping_provider({"providerId": "codex"})

        self.assertEqual(records[0]["id"], "codex")
        self.assertEqual(cached[0]["label"], "Codex")
        self.assertTrue(login["started"])
        self.assertFalse(alias_login["started"])
        self.assertTrue(pending["pending"])
        self.assertEqual(events[-1][0], "assistant_ping_result")
        self.assertEqual(events[-1][1]["statusCode"], 200)

    def test_provider_refresh_falls_back_on_service_error_and_stop_local_model_delegates(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            controller = self._controller(root, _ProviderFacadeService(raise_status=True))
            settings = controller._settings()
            records = controller._refresh_provider_records(settings)

            with patch("infrastructure.local_llm.stop_local_llm", return_value={"stopped": True}) as stop:
                stopped = controller.stop_local_model({"profileId": "fast"})

        self.assertEqual(records[0]["errorCode"], "provider_status_error")
        self.assertTrue(stopped["stopped"])
        stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
