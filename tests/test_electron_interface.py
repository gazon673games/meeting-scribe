from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

from application.event_types import TranscriptSpeakerUpdateEvent, UtteranceEvent
from application.codex_assistant import CodexAssistantResult
from application.codex_config import CodexProfile
from application.diarization_model_download import RECOMMENDED_DIARIZATION_MODELS, diarization_models_dir
from interface.assistant_controller import AssistantController
from interface.backend import ElectronBackend
from interface.jsonl_bridge import JsonLineBridge
from interface.session_controller import HeadlessSessionController
from settings.infrastructure.json_config_repository import JsonConfigRepository
from transcription.application.startup_service import TranscriptionStartupService


class _DeviceCatalog:
    def list_loopback_devices(self) -> List[Tuple[str, object]]:
        return [("Speakers loopback", {"name": "speakers"})]

    def list_input_devices(self) -> List[Tuple[str, int]]:
        return [("Built-in microphone", 7)]


class _FakeSource:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeAudioSourceFactory:
    def create_loopback_source(self, *, name, engine_format, device, error_callback=None):  # noqa: ANN001
        return _FakeSource(name)

    def create_microphone_source(self, *, name, device):  # noqa: ANN001
        return _FakeSource(name)

    def create_process_source(self, *, name, token, error_callback=None):  # noqa: ANN001
        return _FakeSource(name)


class _FakeEngine:
    def __init__(self) -> None:
        self.sources: list[_FakeSource] = []
        self.running = False
        self.tap_queue = None
        self.enabled: dict[str, bool] = {}
        self.delays: dict[str, float] = {}

    def is_running(self) -> bool:
        return self.running

    def set_tap_queue(self, tap_queue) -> None:  # noqa: ANN001
        self.tap_queue = tap_queue

    def set_tap_config(self, **kwargs) -> None:  # noqa: ANN003
        self.tap_config = kwargs

    def add_source(self, src) -> None:  # noqa: ANN001
        self.sources.append(src)
        self.enabled[src.name] = True

    def remove_source(self, name: str) -> None:
        self.sources = [source for source in self.sources if source.name != name]
        self.enabled.pop(name, None)
        self.delays.pop(name, None)

    def add_master_filter(self, flt) -> None:  # noqa: ANN001
        pass

    def set_source_enabled(self, name: str, enabled: bool) -> None:
        self.enabled[name] = bool(enabled)

    def set_source_delay_ms(self, name: str, delay_ms: float) -> None:
        self.delays[name] = float(delay_ms)

    def enable_auto_sync(self, reference_source: str, target_source: str) -> None:
        pass

    def disable_auto_sync(self) -> None:
        pass

    def get_meters(self) -> dict:
        return {
            "master": {"rms": 0.25, "last_ts": 0.0},
            "drops": {"dropped_out_blocks": 0, "dropped_tap_blocks": 0},
            "sources": {
                source.name: {
                    "rms": 0.25,
                    "last_ts": 0.0,
                    "enabled": self.enabled.get(source.name, True),
                    "delay_ms": self.delays.get(source.name, 0.0),
                }
                for source in self.sources
            },
        }

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


class _FakeAudioRuntimeFactory:
    def __init__(self) -> None:
        self.engine = _FakeEngine()

    def create(self, *, format, output_queue, tap_queue=None):  # noqa: ANN001
        return self.engine


class _FakeWriter:
    def __init__(self) -> None:
        self.started = False
        self.recording = False
        self.path = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def start_recording(self, path, fmt) -> None:  # noqa: ANN001
        self.path = path
        self.recording = True

    def stop_recording(self) -> None:
        self.recording = False

    def is_recording(self) -> bool:
        return self.recording

    def target_path(self):
        return self.path

    def last_error(self):
        return None

    def drained_blocks(self) -> int:
        return 0

    def written_blocks(self) -> int:
        return 0


class _FakeWavRecorderFactory:
    def __init__(self) -> None:
        self.writer = _FakeWriter()

    def available(self) -> bool:
        return True

    def create(self, output_queue):  # noqa: ANN001
        return self.writer


class _FakeAssistantService:
    def execute(self, command, *, options, publish_event) -> None:  # noqa: ANN001
        from application.event_types import CodexResultEvent

        publish_event(
            CodexResultEvent(
                ok=True,
                profile=command.profile.label,
                cmd=command.request_text,
                text=f"answer for {command.context_text}",
                dt_s=0.1,
            )
        )


class _FakeAsrRuntime:
    def __init__(self, event_queue) -> None:  # noqa: ANN001
        self.event_queue = event_queue
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True
        self.event_queue.put_nowait(UtteranceEvent(text="hello from asr", stream="mic"))

    def stop(self) -> None:
        self.stopped = True


class _FakeAsrRuntimeFactory:
    def __init__(self) -> None:
        self.runtime = None
        self.settings = None

    def build(self, settings, *, tap_queue, project_root: Path, event_queue=None):  # noqa: ANN001
        self.settings = settings
        self.runtime = _FakeAsrRuntime(event_queue)
        return self.runtime


class ElectronInterfaceTests(unittest.TestCase):
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
            live_source = backend.handle("add_source", {"deviceId": "loopback:0"})
            stopped = backend.handle("stop_session", {})

            self.assertEqual(source["name"], "mic")
            self.assertEqual(live_source["name"], "desktop_audio")
            self.assertEqual(len(runtime_factory.engine.sources), 2)
            self.assertTrue(runtime_factory.engine.running is False)
            self.assertFalse(stopped["running"])
            self.assertTrue(started["wavRecording"])
            self.assertFalse(wav_factory.writer.recording)

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

    def test_headless_session_streams_asr_events(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            runtime_factory = _FakeAudioRuntimeFactory()
            asr_factory = _FakeAsrRuntimeFactory()
            events: list[tuple[str, dict]] = []
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=runtime_factory,
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                asr_runtime_factory=asr_factory,
                transcription_startup_service=TranscriptionStartupService(),
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )

            controller.add_source(kind="input", token=1, label="Mic")
            started = controller.start_session({"asrEnabled": True, "model": "medium", "language": "en"})
            for _ in range(20):
                if any(kind == "transcript_line" for kind, _ in events):
                    break
                time.sleep(0.02)
            snapshot = controller.snapshot()
            controller.stop_session({})

            self.assertTrue(started["asrRunning"])
            self.assertTrue(snapshot["transcript"])
            self.assertEqual(snapshot["transcript"][0]["text"], "hello from asr")
            self.assertEqual(snapshot["transcript"][0]["speaker"], "Me")
            self.assertTrue(asr_factory.runtime.stopped)

    def test_backend_passes_diarization_config_to_asr_session(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "ui": {"asr_enabled": True, "model": "medium", "lang": "en", "asr_mode": 1},
                    "asr": {
                        "diarization_enabled": True,
                        "diar_backend": "sherpa_onnx",
                        "diarization_sidecar_enabled": True,
                        "diarization_queue_size": 12,
                        "diar_sherpa_embedding_model_path": "models/speaker.onnx",
                        "diar_sherpa_provider": "cpu",
                        "diar_sherpa_num_threads": 2,
                    },
                }
            )
            asr_factory = _FakeAsrRuntimeFactory()
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                asr_runtime_factory=asr_factory,
                transcription_startup_service=TranscriptionStartupService(),
            )
            backend = ElectronBackend(root, repository, _DeviceCatalog(), controller)

            backend.handle("list_devices")
            backend.handle("add_source", {"deviceId": "input:0"})
            state = backend.handle("get_state")
            backend.handle("start_session", {})
            backend.handle("stop_session", {"runOfflinePass": False})

            self.assertTrue(state["configSummary"]["diarizationEnabled"])
            self.assertIn("sherpa_onnx", state["options"]["diarizationBackends"])
            self.assertTrue(asr_factory.settings.diarization_enabled)
            self.assertEqual(asr_factory.settings.diar_backend, "sherpa_onnx")
            self.assertEqual(asr_factory.settings.diarization_queue_size, 12)
            self.assertEqual(asr_factory.settings.diar_sherpa_embedding_model_path, "models/speaker.onnx")
            self.assertEqual(asr_factory.settings.diar_sherpa_num_threads, 2)

    def test_backend_uses_cached_sherpa_model_when_online_backend_dependency_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            spec = RECOMMENDED_DIARIZATION_MODELS[0]
            model_dir = diarization_models_dir(root, root / "models")
            model_dir.mkdir(parents=True)
            model_path = model_dir / spec.file_name
            model_path.write_bytes(b"onnx")
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "ui": {"asr_enabled": True, "model": "medium", "lang": "en"},
                    "asr": {
                        "diarization_enabled": True,
                        "diar_backend": "online",
                        "diar_sherpa_embedding_model_path": "",
                    },
                    "models": {"cache_dir": str(root / "models")},
                }
            )
            asr_factory = _FakeAsrRuntimeFactory()
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                asr_runtime_factory=asr_factory,
                transcription_startup_service=TranscriptionStartupService(),
            )
            backend = ElectronBackend(root, repository, _DeviceCatalog(), controller)

            backend.handle("list_devices")
            backend.handle("add_source", {"deviceId": "input:0"})
            with patch("interface.backend._module_available", return_value=False):
                backend.handle("start_session", {})
            backend.handle("stop_session", {"runOfflinePass": False})

            self.assertEqual(asr_factory.settings.diar_backend, "sherpa_onnx")
            self.assertEqual(asr_factory.settings.diar_sherpa_embedding_model_path, str(model_path))

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

    def test_headless_session_applies_post_fact_speaker_update(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            events: list[tuple[str, dict]] = []
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )
            controller.add_source(kind="loopback", token=1, label="Desktop")

            controller._handle_asr_event(  # noqa: SLF001
                UtteranceEvent(
                    text="question",
                    stream="desktop_audio",
                    speaker="Remote",
                    t_start=1.0,
                    t_end=2.0,
                    ts=2.2,
                )
            )
            line_id = controller.snapshot()["transcript"][0]["id"]
            controller._handle_asr_event(  # noqa: SLF001
                TranscriptSpeakerUpdateEvent(
                    line_id=line_id,
                    speaker="Remote S1",
                    confidence=0.9,
                    source="test",
                    ts=2.5,
                )
            )

            line = controller.snapshot()["transcript"][0]
            self.assertEqual(line["speaker"], "Remote S1")
            self.assertEqual(line["speakerSource"], "test")
            self.assertTrue(any(kind == "transcript_line_update" for kind, _ in events))

    def test_headless_session_applies_pending_speaker_update_to_new_line(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            events: list[tuple[str, dict]] = []
            controller = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )
            controller.add_source(kind="loopback", token=1, label="Desktop")

            controller._handle_asr_event(  # noqa: SLF001
                TranscriptSpeakerUpdateEvent(
                    stream="desktop_audio",
                    speaker="Remote S2",
                    t_start=1.0,
                    t_end=2.0,
                    source="test",
                    ts=1.5,
                )
            )
            controller._handle_asr_event(  # noqa: SLF001
                UtteranceEvent(
                    text="answer",
                    stream="desktop_audio",
                    speaker="Remote",
                    t_start=1.0,
                    t_end=2.0,
                    ts=2.2,
                )
            )

            line = controller.snapshot()["transcript"][0]
            self.assertEqual(line["speaker"], "Remote S2")
            self.assertEqual(line["speakerSource"], "test")
            self.assertTrue(any(kind == "transcript_line" for kind, _ in events))

    def test_assistant_controller_uses_session_transcript_context(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "codex": {
                        "enabled": True,
                        "selected_profile": "fast",
                        "profiles": [{"id": "fast", "label": "Fast", "prompt": "help"}],
                    }
                }
            )
            session = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
            )
            session._transcript_lines.append({"ts": 1.0, "stream": "mic", "text": "question"})  # noqa: SLF001
            events: list[tuple[str, dict]] = []
            assistant = AssistantController(
                project_root=root,
                config_repository=repository,
                assistant_service=_FakeAssistantService(),  # type: ignore[arg-type]
                session_controller=session,
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )

            snapshot = assistant.invoke({"requestText": "reply"})
            for _ in range(20):
                if any(kind == "assistant_result" for kind, _ in events):
                    break
                time.sleep(0.02)

            self.assertIn("profiles", snapshot)
            self.assertTrue(any(kind == "assistant_result" for kind, _ in events))
            self.assertIn("question", assistant.snapshot()["lastResponse"]["text"])

    def test_assistant_controller_rejects_empty_transcript_context(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "codex": {
                        "enabled": True,
                        "selected_profile": "fast",
                        "profiles": [{"id": "fast", "label": "Fast", "prompt": "help"}],
                    }
                }
            )
            session = HeadlessSessionController(
                project_root=root,
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=_FakeWavRecorderFactory(),
            )
            events: list[tuple[str, dict]] = []
            assistant = AssistantController(
                project_root=root,
                config_repository=repository,
                assistant_service=_FakeAssistantService(),  # type: ignore[arg-type]
                session_controller=session,
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )

            with self.assertRaisesRegex(RuntimeError, "context is empty"):
                assistant.invoke({"requestText": "reply"})

            self.assertFalse(events)
            self.assertFalse(assistant.snapshot()["busy"])

    def test_assistant_controller_emits_local_model_status_events(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            repository = JsonConfigRepository(root / "config.json")
            repository.write(
                {
                    "codex": {
                        "enabled": True,
                        "selected_profile": "qwen",
                        "profiles": [
                            {
                                "id": "qwen",
                                "label": "qwen",
                                "provider": "openai_local",
                                "model": "local-model",
                                "base_url": "http://127.0.0.1:1234/v1",
                            }
                        ],
                    }
                }
            )
            events: list[tuple[str, dict]] = []
            assistant = AssistantController(
                project_root=root,
                config_repository=repository,
                assistant_service=_FakeAssistantService(),  # type: ignore[arg-type]
                event_sink=lambda typ, payload: events.append((typ, payload)),
            )

            def fake_start(profile, project_root, emit_event):  # noqa: ANN001
                emit_event(
                    {
                        "type": "local_llm_status",
                        "profileId": profile.id,
                        "state": "running",
                        "message": "ready",
                    }
                )
                return {"started": True, "profileId": profile.id}

            with patch("infrastructure.local_llm.start_local_llm_async", side_effect=fake_start):
                result = assistant.start_local_model({"profileId": "qwen"})

            self.assertTrue(result["started"])
            self.assertEqual(len(events), 1)
            kind, payload = events[0]
            self.assertEqual(kind, "local_llm_status")
            self.assertEqual(payload["profileId"], "qwen")
            self.assertEqual(payload["state"], "running")
            self.assertEqual(payload["message"], "ready")
            self.assertIn("ts", payload)

    def test_jsonl_bridge_wraps_success_and_errors(self) -> None:
        stdin = io.StringIO(
            "\n".join(
                [
                    json.dumps({"id": "1", "method": "ping", "params": {"value": 42}}),
                    json.dumps({"id": "2", "method": "missing"}),
                    "",
                ]
            )
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        def handler(method, params):  # noqa: ANN001
            if method == "ping":
                return {"pong": True, "params": params}
            raise KeyError(method)

        JsonLineBridge(handler, stdin=stdin, stdout=stdout, stderr=stderr).serve_forever()
        messages = [json.loads(line) for line in stdout.getvalue().splitlines()]

        self.assertEqual(messages[0]["event"]["type"], "backend_ready")
        self.assertTrue(messages[1]["ok"])
        self.assertEqual(messages[1]["result"]["params"]["value"], 42)
        self.assertFalse(messages[2]["ok"])
        self.assertEqual(messages[2]["error"]["type"], "KeyError")

    def test_jsonl_bridge_preserves_unicode_payloads(self) -> None:
        text = "\u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440"
        reply_prefix = "\u041e\u0442\u0432\u0435\u0442:"
        stdin = io.StringIO(
            json.dumps(
                {"id": "ru", "method": "echo", "params": {"text": text}},
                ensure_ascii=False,
            )
            + "\n"
        )
        stdout = io.StringIO()
        stderr = io.StringIO()

        def handler(method, params):  # noqa: ANN001
            self.assertEqual(method, "echo")
            return {"reply": f"{reply_prefix} {params['text']}"}

        JsonLineBridge(handler, stdin=stdin, stdout=stdout, stderr=stderr).serve_forever()
        raw_output = stdout.getvalue()
        messages = [json.loads(line) for line in raw_output.splitlines()]

        self.assertEqual(messages[1]["result"]["reply"], f"{reply_prefix} {text}")
        self.assertIn(text, raw_output)


if __name__ == "__main__":
    unittest.main()
