from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import List, Tuple

from application.event_types import UtteranceEvent
from application.codex_assistant import CodexAssistantResult
from application.codex_config import CodexProfile
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

    def build(self, settings, *, tap_queue, project_root: Path, event_queue=None):  # noqa: ANN001
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
            stopped = backend.handle("stop_session", {})

            self.assertEqual(source["name"], "mic")
            self.assertTrue(runtime_factory.engine.running is False)
            self.assertFalse(stopped["running"])
            self.assertTrue(started["wavRecording"])
            self.assertFalse(wav_factory.writer.recording)

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
            self.assertTrue(asr_factory.runtime.stopped)

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


if __name__ == "__main__":
    unittest.main()
