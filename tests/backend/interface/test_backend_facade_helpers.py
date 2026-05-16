from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from application.codex_config import CodexSettings
from interface import backend as backend_facade
from interface import backend_impl
from interface.assistant_controller_parts.request_plan import build_request_plan
from interface.backend_parts.session_mixin import BackendSessionMixin
from interface.backend_parts.session_orchestration import download_then_start, resolve_diarization_start_params
from interface.backend_parts.state_mixin import _ui_model
from interface.backend_parts.system_utils import module_available
from interface.session_controller import HeadlessSessionController
from settings.infrastructure.json_config_repository import JsonConfigRepository
from tests.helpers.electron_interface_fakes import (
    _DeviceCatalog,
    _FakeAudioRuntimeFactory,
    _FakeAudioSourceFactory,
    _FakeWavRecorderFactory,
)


class BackendFacadeHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_keys = ("TMP", "TEMP", "TMPDIR")
        self._old_env = {key: os.environ.get(key) for key in self._env_keys}
        self._old_tempdir = tempfile.tempdir

    def tearDown(self) -> None:
        for key, value in self._old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = self._old_tempdir

    def test_backend_base_state_and_session_delegates_cover_thin_facades(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write({"ui": {"model": "tiny"}})
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
            )
            backend = backend_facade.ElectronBackend(root, repository, _DeviceCatalog(), controller)
            emitted: list[tuple[str, dict]] = []
            assistant = SimpleNamespace(
                invoke=lambda params: {"invoked": params},
                start_login=lambda params: {"login": params},
                ping_provider=lambda params: {"ping": params},
                start_local_model=lambda params: {"startLocal": params},
                stop_local_model=lambda params: {"stopLocal": params},
                set_event_sink=lambda sink: setattr(assistant, "sink", sink),
                snapshot=lambda: {"enabled": True},
            )
            backend.assistant_controller = assistant

            backend._event_sink = lambda event_type, payload: emitted.append((event_type, payload))
            backend._emit("unit_event", {"value": 1})
            self.assertEqual(emitted[0][0], "unit_event")
            self.assertEqual(backend._models_dir_from_params({"modelsDir": str(root / "models2")}), (root / "models2").resolve())
            self.assertEqual(backend.ping({"x": 1})["echo"], {"x": 1})
            self.assertEqual(backend.get_config()["ui"]["model"], "tiny")
            self.assertEqual(_ui_model(repository.read()), "tiny")

            backend.set_event_sink(lambda event_type, payload: emitted.append((event_type, payload)))
            self.assertIs(controller.event_sink, backend._event_sink)
            self.assertIs(assistant.sink, backend._event_sink)

            backend.handle("list_devices")
            backend.handle("add_source", {"deviceId": "input:0"})
            delayed = backend.handle("set_source_delay", {"name": "mic", "delayMs": -5})
            self.assertEqual(delayed["delayMs"], 0.0)
            self.assertEqual(backend.handle("invoke_assistant", {"request": "x"})["invoked"]["request"], "x")
            self.assertIn("login", backend.handle("start_assistant_login", {"providerId": "codex"}))
            self.assertIn("ping", backend.handle("ping_assistant_provider", {"providerId": "codex"}))
            self.assertIn("startLocal", backend.handle("start_local_llm", {"profileId": "p"}))
            self.assertIn("stopLocal", backend.handle("stop_local_llm", {"profileId": "p"}))

            state = backend.handle("save_config", {"config": {"models": {"cache_dir": str(root / "models")}}})
            self.assertEqual(state["paths"]["models"], str((root / "models").resolve()))
            self.assertIs(BackendSessionMixin._resolve_diarization_start_params(backend, {"diarizationEnabled": False})["diarizationEnabled"], False)

    def test_diarization_resolution_module_checks_gpu_cache_and_request_plans(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            owner = SimpleNamespace(project_root=root, _models_dir=lambda config=None: root / "models")
            cached = {"path": str(root / "speaker.onnx"), "provider": "cpu"}
            with (
                patch("interface.backend_parts.session_orchestration.module_available", return_value=False),
                patch("application.diarization_model_download.default_cached_diarization_model", return_value=cached),
            ):
                params = resolve_diarization_start_params(owner, {"diarizationEnabled": True, "diarBackend": "online"})

            self.assertEqual(params["diarBackend"], "sherpa_onnx")
            self.assertTrue(module_available("sys"))
            self.assertTrue(backend_facade._module_available("sys"))
            self.assertEqual(build_request_plan({"action": "summary"}, CodexSettings()).source_label, "summary")

            repository = JsonConfigRepository(root / "config.json")
            repository.write({})
            impl = backend_impl.ElectronBackend(root, repository, _DeviceCatalog())
            with patch("interface.backend_impl._nvidia_gpu_snapshot", return_value=[{"name": "GPU"}]) as snapshot:
                self.assertEqual(impl._gpu_snapshot(max_age_s=10)[0]["name"], "GPU")
                self.assertEqual(impl._gpu_snapshot(max_age_s=10)[0]["name"], "GPU")
            snapshot.assert_called_once()

    def test_download_then_start_downloads_missing_model_before_starting_session(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            started = []
            finished = []
            owner = SimpleNamespace(
                _models_dir=lambda: root / "models",
                download_model=lambda params: setattr(owner, "downloaded", params["name"]),
                _download_record=lambda name: {"state": "done"},
            )
            controller = SimpleNamespace(
                begin_model_download=lambda name: setattr(controller, "begun", name),
                finish_model_download=lambda error: finished.append(error),
                start_session=lambda params: started.append(params) or {"started": True},
            )

            with (
                patch("application.model_download.normalize_model_reference", return_value="openai/whisper-tiny"),
                patch("application.model_download.is_model_cached", return_value=False),
            ):
                result = download_then_start(owner, controller, {"model": "tiny", "downloadWaitTimeoutS": 0.1})

            self.assertTrue(result["started"])
            self.assertEqual(controller.begun, "openai/whisper-tiny")
            self.assertEqual(owner.downloaded, "openai/whisper-tiny")
            self.assertEqual(finished, [""])


if __name__ == "__main__":
    unittest.main()
