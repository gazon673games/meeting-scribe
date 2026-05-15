from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from application.asr_profiles import PROFILE_ULTRA_FAST
from interface.backend import ElectronBackend
from interface.session_controller import HeadlessSessionController
from settings.infrastructure.json_config_repository import JsonConfigRepository
from tests.helpers.electron_interface_fakes import (
    _DeviceCatalog,
    _FakeAudioRuntimeFactory,
    _FakeAudioSourceFactory,
    _FakeWavRecorderFactory,
)


class ElectronInterfaceBackendCoreTests(unittest.TestCase):
    def test_backend_returns_state_config_and_devices(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "ui": {
                        "lang": "ru",
                        "model": "medium",
                        "profile": "Balanced",
                        "asr_enabled": True,
                        "asr_mode": 1,
                    },
                    "asr": {"compute_type": "float16"},
                    "codex": {"enabled": True, "profiles": [{"id": "fast"}]},
                }
            )
            backend = ElectronBackend(root, repository, _DeviceCatalog())

            state = backend.handle("get_state")
            devices = backend.handle("list_devices")

            self.assertEqual(state["configSummary"]["language"], "ru")
            self.assertEqual(state["configSummary"]["asrMode"], "split")
            self.assertFalse(state["capabilities"]["sessionControl"])
            self.assertEqual(state["options"]["asrProfiles"][0], PROFILE_ULTRA_FAST)
            self.assertIn(PROFILE_ULTRA_FAST, state["options"]["streamingLockedProfiles"])
            self.assertTrue(state["options"]["profileDefaults"][PROFILE_ULTRA_FAST]["streaming_enabled"])
            self.assertEqual(devices["loopback"][0]["id"], "loopback:0")
            self.assertEqual(devices["input"][0]["label"], "Built-in microphone")

    def test_backend_adds_source_and_controls_headless_session(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write({"ui": {"asr_enabled": False, "wav_enabled": True, "output_file": "test.wav"}})
            runtime_factory = _FakeAudioRuntimeFactory()
            wav_factory = _FakeWavRecorderFactory()
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=runtime_factory,
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=wav_factory,
            )
            backend = ElectronBackend(root, repository, _DeviceCatalog(), controller)

            backend.handle("list_devices")
            source = backend.handle("add_source", {"deviceId": "input:0"})
            started = backend.handle("start_session", {})
            self.assertTrue(runtime_factory.engine.output_enabled)
            live_source = backend.handle("add_source", {"deviceId": "loopback:0"})
            stopped = backend.handle("stop_session", {})

            self.assertEqual(source["name"], "mic")
            self.assertEqual(live_source["name"], "desktop_audio")
            self.assertEqual(len(runtime_factory.engine.sources), 2)
            self.assertTrue(runtime_factory.engine.running is False)
            self.assertFalse(stopped["running"])
            self.assertTrue(started["wavRecording"])
            self.assertFalse(runtime_factory.engine.output_enabled)
            self.assertFalse(wav_factory.writer.recording)

    def test_runtime_state_is_lightweight_and_resource_usage_can_skip_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write({"ui": {"asr_enabled": False, "wav_enabled": False}})
            backend = ElectronBackend(root, repository, _DeviceCatalog())

            runtime_state = backend.handle("get_runtime_state")
            self.assertIn("session", runtime_state)
            self.assertNotIn("assistant", runtime_state)
            self.assertNotIn("hardware", runtime_state)

            with patch("interface.backend._nvidia_gpu_snapshot") as gpu_snapshot:
                usage = backend.handle("get_resource_usage", {"includeGpu": False})
                self.assertEqual(usage["gpus"], [])
                gpu_snapshot.assert_not_called()

    def test_backend_lists_grouped_process_sessions_and_adds_process_source(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write({})
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
            )
            backend = ElectronBackend(root, repository, _DeviceCatalog(), controller)

            grouped_sessions = [
                {
                    "id": "endpoint-headphones",
                    "label": "Headphones",
                    "sessions": [{"pid": 1234, "label": "Browser", "streams": 1, "endpointId": "endpoint-headphones"}],
                }
            ]

            with (
                patch("infrastructure.process_session_catalog.is_per_process_audio_supported", return_value=True),
                patch("infrastructure.process_session_catalog.list_process_session_groups", return_value=grouped_sessions),
            ):
                catalog = backend.handle("list_process_sessions")
                source = backend.handle("add_source", {"deviceId": catalog["sessions"][0]["id"]})

            self.assertEqual(catalog["groups"][0]["label"], "Headphones")
            self.assertEqual(catalog["groups"][0]["sessions"][0]["fullLabel"], "Headphones / Browser")
            self.assertEqual(source["kind"], "process")
            self.assertEqual(source["label"], "Headphones / Browser")

    def test_backend_keeps_process_tokens_when_refreshing_devices(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write({})
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
            )
            backend = ElectronBackend(root, repository, _DeviceCatalog(), controller)

            grouped_sessions = [
                {
                    "id": "endpoint-speakers",
                    "label": "Speakers",
                    "sessions": [{"pid": 4321, "label": "Player", "streams": 1, "endpointId": "endpoint-speakers"}],
                }
            ]

            with (
                patch("infrastructure.process_session_catalog.is_per_process_audio_supported", return_value=True),
                patch("infrastructure.process_session_catalog.list_process_session_groups", return_value=grouped_sessions),
            ):
                catalog = backend.handle("list_process_sessions")
                backend.handle("list_devices")
                source = backend.handle("add_source", {"deviceId": catalog["sessions"][0]["id"]})

            self.assertEqual(source["kind"], "process")
            self.assertEqual(source["label"], "Speakers / Player")

    def test_backend_removes_source_and_clears_transcript_when_idle(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write({})
            runtime_factory = _FakeAudioRuntimeFactory()
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=runtime_factory,
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
            )
            backend = ElectronBackend(root, repository, _DeviceCatalog(), controller)

            backend.handle("list_devices")
            backend.handle("add_source", {"deviceId": "input:0"})
            controller._transcript_lines.append({"ts": 1.0, "stream": "mic", "text": "question"})  # noqa: SLF001
            removed = backend.handle("remove_source", {"name": "mic"})
            cleared = backend.handle("clear_transcript")

            self.assertEqual(removed["name"], "mic")
            self.assertEqual(runtime_factory.engine.sources, [])
            self.assertEqual(cleared["transcript"], [])

    def test_backend_lists_local_llm_gguf_models(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            model_dir = root / "models" / "llm" / "llama3"
            model_dir.mkdir(parents=True)
            model_path = model_dir / "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
            model_path.write_bytes(b"gguf")
            repository = JsonConfigRepository(root / "config.json")
            repository.write({})
            backend = ElectronBackend(root, repository, _DeviceCatalog())

            result = backend.handle("list_llm_models", {})

            self.assertEqual(result["models"][0]["label"], model_path.name)
            self.assertEqual(result["models"][0]["modelAlias"], "Meta-Llama-3-8B-Instruct-Q4_K_M")

