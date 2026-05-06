from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from application.event_types import UtteranceEvent


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
        self.output_enabled = False
        self.enabled: dict[str, bool] = {}
        self.delays: dict[str, float] = {}

    def is_running(self) -> bool:
        return self.running

    def set_tap_queue(self, tap_queue) -> None:  # noqa: ANN001
        self.tap_queue = tap_queue

    def set_output_enabled(self, enabled: bool) -> None:
        self.output_enabled = bool(enabled)

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

