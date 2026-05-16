from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from interface.session_controller import HeadlessSessionController
from tests.helpers.electron_interface_fakes import (
    _FakeAudioRuntimeFactory,
    _FakeAudioSourceFactory,
    _FakeWavRecorderFactory,
)


class _OfflinePassUseCase:
    def __init__(self, *, available: bool = True, error: Exception | None = None) -> None:
        self._available = available
        self.error = error
        self.requests = []

    def available(self) -> bool:
        return self._available

    def execute(self, request):  # noqa: ANN001
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(out_txt="offline transcript")


class _ImmediateThread:
    def __init__(self, *, target, args, name, daemon) -> None:  # noqa: ANN001
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon

    def start(self) -> None:
        self.target(*self.args)


def _controller(root: Path, offline_pass_use_case=None, event_sink=None):  # noqa: ANN001
    return HeadlessSessionController(
        project_root=root,
        audio_runtime_factory=_FakeAudioRuntimeFactory(),
        audio_source_factory=_FakeAudioSourceFactory(),
        wav_recorder_factory=_FakeWavRecorderFactory(),
        offline_pass_use_case=offline_pass_use_case,
        event_sink=event_sink,
    )


class SessionRuntimeMixinTests(unittest.TestCase):
    def test_source_error_event_and_shutdown_close_runtime_resources(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            events: list[tuple[str, dict]] = []
            wav_factory = _FakeWavRecorderFactory()
            controller = HeadlessSessionController(
                project_root=Path(raw_root),
                audio_runtime_factory=_FakeAudioRuntimeFactory(),
                audio_source_factory=_FakeAudioSourceFactory(),
                wav_recorder_factory=wav_factory,
                event_sink=lambda kind, payload: events.append((kind, payload)),
            )

            controller.add_source(kind="input", token=7, label="Mic")
            controller._on_source_error("mic", "device lost")  # noqa: SLF001
            controller.shutdown()
            controller.shutdown()

            self.assertFalse(wav_factory.writer.started)
            self.assertEqual(controller.snapshot()["lastError"], "mic: device lost")
            self.assertEqual(events[-1][0], "source_error")
            self.assertEqual(events[-1][1]["source"], "mic")

    def test_model_download_lifecycle_updates_state_progress_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            events: list[tuple[str, dict]] = []
            controller = _controller(Path(raw_root), event_sink=lambda kind, payload: events.append((kind, payload)))

            controller.begin_model_download("large-v3")
            controller.update_model_download_progress({"downloadedBytes": "2048", "speedBps": "128.5", "message": "Halfway"})
            progress_snapshot = controller.snapshot()["modelDownload"]
            controller.finish_model_download("network down")

            self.assertEqual(progress_snapshot, {"model": "large-v3", "downloadedBytes": 2048, "speedBps": 128.5, "message": "Halfway"})
            self.assertEqual(controller.snapshot()["modelDownload"], {})
            self.assertEqual(controller.snapshot()["lastError"], "network down")
            self.assertIn(("session_error", {"ts": events[1][1]["ts"], "message": "network down"}), events)
            self.assertEqual(events[-1][0], "session_state_changed")
            self.assertEqual(events[-1][1]["state"], "idle")

    def test_offline_pass_runs_after_stop_and_records_success(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            wav_path = root / "capture.wav"
            wav_path.write_bytes(b"wav")
            events: list[tuple[str, dict]] = []
            use_case = _OfflinePassUseCase()
            controller = _controller(root, offline_pass_use_case=use_case, event_sink=lambda kind, payload: events.append((kind, payload)))

            self.assertFalse(controller._should_run_offline_pass({"runOfflinePass": True}, root / "missing.wav"))  # noqa: SLF001
            self.assertTrue(controller._should_run_offline_pass({"runOfflinePass": True}, wav_path))  # noqa: SLF001

            with patch("interface.session_controller_parts.runtime_offline_pass_mixin.threading.Thread", _ImmediateThread):
                controller._start_offline_pass(wav_path, {"model": "large-v3", "language": "en"})  # noqa: SLF001

            snapshot = controller.snapshot()["offlinePass"]
            self.assertFalse(snapshot["running"])
            self.assertEqual(snapshot["result"]["status"], "done")
            self.assertEqual(snapshot["result"]["outTxt"], "offline transcript")
            self.assertEqual(use_case.requests[0].model_name, "large-v3")
            self.assertEqual(use_case.requests[0].language, "en")
            self.assertEqual([kind for kind, _ in events], ["offline_pass_started", "offline_pass_done"])

    def test_offline_pass_records_errors_and_ignores_duplicate_starts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            wav_path = root / "capture.wav"
            wav_path.write_bytes(b"wav")
            events: list[tuple[str, dict]] = []
            use_case = _OfflinePassUseCase(error=RuntimeError("decode failed"))
            controller = _controller(root, offline_pass_use_case=use_case, event_sink=lambda kind, payload: events.append((kind, payload)))

            controller._offline_pass_running = True  # noqa: SLF001
            controller._start_offline_pass(wav_path, {})  # noqa: SLF001
            self.assertEqual(events, [])
            controller._offline_pass_running = False  # noqa: SLF001

            with patch("interface.session_controller_parts.runtime_offline_pass_mixin.threading.Thread", _ImmediateThread):
                controller._start_offline_pass(wav_path, {})  # noqa: SLF001

            snapshot = controller.snapshot()["offlinePass"]
            self.assertFalse(snapshot["running"])
            self.assertEqual(snapshot["result"]["status"], "error")
            self.assertIn("decode failed", snapshot["result"]["error"])
            self.assertIn("decode failed", controller.snapshot()["lastError"])
            self.assertEqual(events[-1][0], "offline_pass_error")


if __name__ == "__main__":
    unittest.main()
