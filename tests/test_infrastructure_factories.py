from __future__ import annotations

import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from application.offline_pass import OfflineAsrRequest
from audio.domain.formats import AudioFormat
from infrastructure.audio_runtime import DefaultAudioRuntimeFactory
from infrastructure.audio_source_factory import DefaultAudioSourceFactory
from infrastructure.background_tasks import ThreadBackgroundTaskRunner
from infrastructure.device_catalog import SoundDeviceCatalog
from infrastructure.offline_asr import FasterWhisperOfflineAsrRunner
from infrastructure.wav_recording import WavWriterFactory


class _Source:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.args = args
        self.kwargs = kwargs
        self.error_callback = None

    def set_error_callback(self, callback) -> None:  # noqa: ANN001
        self.error_callback = callback


class InfrastructureFactoriesTests(unittest.TestCase):
    def test_runtime_catalog_wav_and_background_factories_delegate_to_infrastructure(self) -> None:
        fmt = AudioFormat(sample_rate=48_000, channels=2)
        output_q: queue.Queue = queue.Queue()
        tap_q: queue.Queue = queue.Queue()

        with patch("infrastructure.audio_runtime.AudioEngine", return_value="engine") as engine:
            self.assertEqual(DefaultAudioRuntimeFactory().create(format=fmt, output_queue=output_q, tap_queue=tap_q), "engine")
        engine.assert_called_once_with(format=fmt, output_queue=output_q, tap_queue=tap_q)

        with (
            patch("infrastructure.device_catalog._list_loopback_devices", return_value=[("speaker", object())]),
            patch("infrastructure.device_catalog._list_input_devices", return_value=[("mic", 1)]),
        ):
            catalog = SoundDeviceCatalog()
            self.assertEqual(catalog.list_loopback_devices()[0][0], "speaker")
            self.assertEqual(catalog.list_input_devices(), [("mic", 1)])

        with (
            patch("infrastructure.wav_recording.soundfile_available", return_value=True),
            patch("infrastructure.wav_recording.WavWriterThread", return_value="writer") as writer,
        ):
            factory = WavWriterFactory()
            self.assertTrue(factory.available())
            self.assertEqual(factory.create(output_q), "writer")
        writer.assert_called_once_with(output_q)

        done = threading.Event()
        handle = ThreadBackgroundTaskRunner().start(name="unit-task", target=lambda value: done.set(), args=(1,))
        handle.join(timeout=1.0)
        self.assertTrue(done.is_set())

    def test_audio_source_factory_creates_loopback_microphone_and_process_sources(self) -> None:
        fmt = AudioFormat(sample_rate=48_000, channels=2)
        factory = DefaultAudioSourceFactory()
        callback = Mock()

        with patch.dict(
            "sys.modules",
            {"audio.infrastructure.sources.wasapi_loopback": SimpleNamespace(WasapiLoopbackSource=_Source)},
        ):
            loopback = factory.create_loopback_source(name="speaker", engine_format=fmt, device="dev", error_callback=callback)
        self.assertEqual(loopback.kwargs["name"], "speaker")
        self.assertIs(loopback.error_callback, callback)

        with patch.dict(
            "sys.modules",
            {"audio.infrastructure.sources.microphone": SimpleNamespace(MicrophoneSource=_Source)},
        ):
            mic = factory.create_microphone_source(name="mic", device="2")
        self.assertEqual(mic.kwargs["format"].channels, 1)
        self.assertEqual(mic.kwargs["device"], 2)

        with (
            patch.object(sys, "platform", "win32"),
            patch.dict(
                "sys.modules",
                {"audio.infrastructure.sources.process_loopback_win": SimpleNamespace(ProcessLoopbackWinSource=_Source)},
            ),
        ):
            win_source = factory.create_process_source(name="app", token={"pid": 123}, error_callback=callback)
        self.assertEqual(win_source.kwargs["pid"], 123)
        self.assertIs(win_source.error_callback, callback)

        with (
            patch.object(sys, "platform", "linux"),
            patch.dict(
                "sys.modules",
                {"audio.infrastructure.sources.pulse_app_source": SimpleNamespace(PulseAppSource=_Source)},
            ),
        ):
            linux_source = factory.create_process_source(name="app", token={"index": 7})
        self.assertEqual(linux_source.kwargs["sink_input_index"], 7)

    def test_offline_asr_runner_reports_availability_and_delegates_run(self) -> None:
        runner = FasterWhisperOfflineAsrRunner()

        with (
            patch("infrastructure.offline_asr.OfflineRunner", None),
            patch("infrastructure.offline_asr.OfflineProfile", None),
        ):
            self.assertFalse(runner.available())
            with self.assertRaises(RuntimeError):
                runner.run(
                    OfflineAsrRequest(
                        project_root=Path("."),
                        wav_path=Path("in.wav"),
                        out_txt=Path("out.txt"),
                        model_name="large-v3",
                        language="ru",
                    )
                )

        fake_runner = Mock()
        fake_runner.run.return_value = "out.txt"
        offline_runner_cls = Mock(return_value=fake_runner)
        offline_profile_cls = Mock(return_value="profile")
        with (
            patch("infrastructure.offline_asr.OfflineRunner", offline_runner_cls),
            patch("infrastructure.offline_asr.OfflineProfile", offline_profile_cls),
        ):
            self.assertTrue(runner.available())
            result = runner.run(
                OfflineAsrRequest(
                    project_root=Path("project"),
                    wav_path=Path("in.wav"),
                    out_txt=Path("out.txt"),
                    model_name="tiny",
                    language="en",
                )
            )

        self.assertEqual(result, Path("out.txt"))
        offline_runner_cls.assert_called_once_with(project_root=Path("project"))
        offline_profile_cls.assert_called_once_with(model_name="tiny", language="en")
        fake_runner.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
