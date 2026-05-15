from __future__ import annotations

import queue
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from audio.application.engine import AudioEngine
from audio.domain.formats import AudioFormat
from audio.infrastructure import devices, writer


class _FakeThread:
    def __init__(self, *, target, name="", daemon=False) -> None:  # noqa: ANN001
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        self.joined = False

    def start(self) -> None:
        self.started = True

    def join(self, timeout=None) -> None:  # noqa: ANN001
        self.joined = True


class _Source:
    def __init__(self, name: str, *, fail_start: bool = False) -> None:
        self.name = name
        self.fail_start = fail_start
        self.callback = None
        self.started = 0
        self.stopped = 0

    def start(self, on_audio) -> None:  # noqa: ANN001
        if self.fail_start:
            raise RuntimeError("start failed")
        self.callback = on_audio
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def get_format(self) -> AudioFormat:
        return AudioFormat(sample_rate=48_000, channels=1, blocksize=2)

    def get_filters(self) -> list:
        return []


class _StopAfterOne:
    def __init__(self) -> None:
        self.calls = 0
        self.was_set = False

    def is_set(self) -> bool:
        self.calls += 1
        return self.calls > 1

    def set(self) -> None:
        self.was_set = True


class _OneFrameQueue:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame
        self.used = False

    def get(self, timeout=None):  # noqa: ANN001, ANN201
        if self.used:
            raise queue.Empty()
        self.used = True
        return self.frame


class _WavFile:
    def __init__(self, *, fail_write: bool = False) -> None:
        self.fail_write = fail_write
        self.writes = 0
        self.closed = False

    def write(self, frame) -> None:  # noqa: ANN001
        if self.fail_write:
            raise RuntimeError("disk full")
        self.writes += 1

    def close(self) -> None:
        self.closed = True


class AudioEngineAndInfrastructureTests(unittest.TestCase):
    def test_engine_start_buffers_audio_and_reports_runtime_meters(self) -> None:
        fmt = AudioFormat(sample_rate=48_000, channels=2, blocksize=2)
        engine = AudioEngine(fmt, queue.Queue(), max_source_buffer_blocks=1, tap_queue=queue.Queue(), tap_queue_max=5)

        with self.assertRaises(RuntimeError):
            engine.start()

        source = _Source("mic")
        engine.add_source(source)
        engine.set_tap_queue(queue.Queue())
        engine.set_tap_config(mode="sources", sources=["mic"], drop_threshold=0.2)
        engine.set_output_enabled(False)
        engine.enable_auto_sync("speaker", "mic")
        engine.disable_auto_sync()
        engine.add_master_filter(Mock(process=lambda block, fmt: block))

        with patch("audio.application.engine.threading.Thread", _FakeThread):
            engine.start()

        self.assertTrue(engine.is_running())
        self.assertEqual(source.started, 1)
        assert source.callback is not None
        source.callback("mic", np.array([0.5, 0.25], dtype=np.float32))

        snapshot = engine._build_mix_snapshot(0.1)
        self.assertTrue(snapshot.running)
        self.assertEqual(snapshot.tap_mode, "sources")
        self.assertEqual(snapshot.tap_sources_filter, {"mic"})
        self.assertFalse(snapshot.output_enabled)

        engine._record_master_metrics(0.5, 12.0)
        engine._record_output_drop()
        engine._record_tap_drop()
        meters = engine.get_meters()
        self.assertEqual(meters["master"]["rms"], 0.5)
        self.assertEqual(meters["drops"]["dropped_out_blocks"], 1)
        self.assertEqual(meters["drops"]["dropped_tap_blocks"], 1)

        engine.set_source_enabled("mic", False)
        source.callback("mic", np.array([1.0], dtype=np.float32))
        engine.stop()
        engine.stop()

        self.assertFalse(engine.is_running())
        self.assertEqual(source.stopped, 1)
        self.assertFalse(engine._build_mix_snapshot(0.1).running)

    def test_engine_rolls_back_when_initial_or_late_source_start_fails(self) -> None:
        fmt = AudioFormat(sample_rate=48_000, channels=1, blocksize=2)
        engine = AudioEngine(fmt, queue.Queue())
        good = _Source("good")
        bad = _Source("bad", fail_start=True)
        engine.add_source(good)
        engine.add_source(bad)

        with self.assertRaises(RuntimeError):
            engine.start()
        self.assertFalse(engine.is_running())
        self.assertEqual(good.stopped, 1)

        engine = AudioEngine(fmt, queue.Queue())
        engine.add_source(good := _Source("good"))
        with patch("audio.application.engine.threading.Thread", _FakeThread):
            engine.start()
        with self.assertRaises(RuntimeError):
            engine.add_source(_Source("late", fail_start=True))
        self.assertNotIn("late", [name for name, _state in engine._registry.source_items()])
        engine.stop()

    def test_device_discovery_prefers_loopbacks_and_filters_input_devices(self) -> None:
        mic_loop = SimpleNamespace(name="Default Speaker Loopback", id="loop", isloopback=True)
        mic_plain = SimpleNamespace(name="Plain Mic", id=2, isloopback=False)

        self.assertEqual(devices._collect_loopback_candidates([mic_loop, mic_plain]), [("Default Speaker Loopback", "loop")])
        self.assertEqual(devices._collect_loopback_candidates([mic_plain]), [("Plain Mic", 2)])

        soundcard = SimpleNamespace(
            all_microphones=lambda include_loopback=True: [mic_plain, mic_loop, mic_loop],
            default_speaker=lambda: SimpleNamespace(name="Default Speaker"),
        )
        sounddevice = SimpleNamespace(
            query_devices=lambda: [
                {"name": "Mic", "max_input_channels": 2, "default_samplerate": 48_000},
                {"name": "Out", "max_input_channels": 0, "default_samplerate": 48_000},
                object(),
            ]
        )
        with patch.dict("sys.modules", {"soundcard": soundcard, "sounddevice": sounddevice}):
            self.assertEqual(devices.list_loopback_devices(), [("Default Speaker Loopback", "loop")])
            self.assertEqual(devices.list_input_devices(), [("[0] Mic (in=2, sr=48000)", 0)])

        with patch.dict("sys.modules", {"soundcard": None, "sounddevice": None}):
            self.assertEqual(devices.list_loopback_devices(), [])
            self.assertEqual(devices.list_input_devices(), [])

    def test_wav_writer_writes_frames_and_stops_recording_on_write_error(self) -> None:
        fmt = AudioFormat(sample_rate=16_000, channels=1)

        with patch("audio.infrastructure.writer.sf", None):
            self.assertFalse(writer.soundfile_available())
            with self.assertRaises(RuntimeError):
                writer.WavWriterThread(queue.Queue()).start_recording(Mock(), fmt)

        wav_file = _WavFile()
        with patch("audio.infrastructure.writer.sf", SimpleNamespace(SoundFile=lambda *args, **kwargs: wav_file)):
            wav_writer = writer.WavWriterThread(_OneFrameQueue(np.zeros((2, 1), dtype=np.float32)))
            wav_writer._stop = _StopAfterOne()  # type: ignore[method-assign]
            wav_writer.start_recording(Path("out.wav"), fmt)
            self.assertTrue(wav_writer.is_recording())
            wav_writer.run()
            self.assertEqual(wav_writer.written_blocks(), 1)
            self.assertEqual(wav_writer.drained_blocks(), 1)
            wav_writer.stop_recording()
            self.assertFalse(wav_writer.is_recording())

        bad_wav = _WavFile(fail_write=True)
        with patch("audio.infrastructure.writer.sf", SimpleNamespace(SoundFile=lambda *args, **kwargs: bad_wav)):
            wav_writer = writer.WavWriterThread(_OneFrameQueue(np.zeros((2, 1), dtype=np.float32)))
            wav_writer._stop = _StopAfterOne()  # type: ignore[method-assign]
            wav_writer.start_recording(Path("out.wav"), fmt)
            wav_writer.run()
            self.assertIn("RuntimeError", wav_writer.last_error() or "")
            self.assertFalse(wav_writer.is_recording())

            wav_writer.join = Mock()  # type: ignore[method-assign]
            wav_writer.stop()
            self.assertTrue(wav_writer._stop.was_set)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
