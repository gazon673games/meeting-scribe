from __future__ import annotations

import queue
import unittest
from dataclasses import dataclass
from threading import Event

import numpy as np

from audio.application.engine_runtime import EngineRuntimeState
from audio.application.mix_worker import AudioMixWorker, MixLoopSnapshot
from audio.application.source_controls import (
    reset_source_runtime_state,
    set_source_delay,
    set_source_enabled_state,
)
from audio.application.source_registry import SourceRegistry
from audio.application.source_state import SourceState
from audio.application.tap_config import normalize_tap_config
from audio.domain.formats import AudioFormat


@dataclass
class _FakeSource:
    name: str = "fake"

    def start(self, on_audio):  # noqa: ANN001
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def get_format(self) -> AudioFormat:
        return AudioFormat(sample_rate=48_000, channels=2, blocksize=480)

    def get_filters(self) -> list:
        return []


class AudioApplicationControlsTests(unittest.TestCase):
    def test_normalize_tap_config_falls_back_and_clamps_threshold(self) -> None:
        config = normalize_tap_config(
            mode="unexpected",  # type: ignore[arg-type]
            sources=["mic", "speaker", "mic"],
            drop_threshold=4.0,
        )

        self.assertEqual(config.mode, "both")
        self.assertEqual(config.sources_filter, {"mic", "speaker"})
        self.assertEqual(config.drop_threshold, 0.99)

    def test_source_controls_reset_buffers_and_delay_without_engine(self) -> None:
        state = SourceState(src=_FakeSource())
        state.buf.append("frame")  # type: ignore[arg-type]
        state.delay_buf.append("delayed")  # type: ignore[arg-type]
        state.buffer_frames = 256

        set_source_delay(state, AudioFormat(sample_rate=48_000, channels=2), 25.0)
        set_source_enabled_state(state, False)

        self.assertFalse(state.enabled)
        self.assertEqual(state.delay_frames, 1200)
        self.assertEqual(len(state.buf), 0)
        self.assertEqual(len(state.delay_buf), 0)
        self.assertEqual(state.buffer_frames, 0)
        self.assertEqual(state.rms, 0.0)
        self.assertGreater(state.last_ts, 0.0)

        state.dropped_in_frames = 10
        state.missing_out_frames = 5
        reset_source_runtime_state(state)

        self.assertEqual(state.dropped_in_frames, 0)
        self.assertEqual(state.missing_out_frames, 0)
        self.assertEqual(state.last_ts, 0.0)

    def test_mix_worker_emits_one_mixed_block(self) -> None:
        fmt = AudioFormat(sample_rate=48_000, channels=2, blocksize=480)
        out_q: queue.Queue[np.ndarray] = queue.Queue()
        stop_evt = Event()
        state = SourceState(src=_FakeSource())
        state.buf.append(np.full((480, 2), 0.25, dtype=np.float32))
        recorded_master: list[float] = []

        def snapshot_provider(period_s: float) -> MixLoopSnapshot:
            return MixLoopSnapshot(
                running=True,
                t_start=0.0,
                t_end=period_s,
                items=[("fake", state)],
                master_filters=[],
                tap_q=None,
            )

        def record_master(master_rms: float, ts_mono: float) -> None:
            recorded_master.append(master_rms)
            stop_evt.set()

        worker = AudioMixWorker(
            format=fmt,
            output_queue=out_q,
            stop_event=stop_evt,
            tap_queue_max=10,
            active_rms_eps=1e-4,
            snapshot_provider=snapshot_provider,
            record_master_metrics=record_master,
            record_output_drop=lambda: None,
            record_tap_drop=lambda: None,
        )

        worker.run()

        mixed = out_q.get_nowait()
        self.assertEqual(mixed.shape, (480, 2))
        np.testing.assert_allclose(mixed, 0.25)
        self.assertTrue(recorded_master)

    def test_source_registry_owns_source_state_commands(self) -> None:
        registry = SourceRegistry(AudioFormat(sample_rate=48_000, channels=2))
        source = _FakeSource(name="mic")

        registry.add_source(source, running=False)
        registry.set_source_delay_ms("mic", 10.0)
        registry.set_source_enabled("mic", False)

        state = registry.get_state("mic")
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.delay_frames, 480)
        self.assertFalse(state.enabled)
        self.assertEqual(registry.sources, [source])
        self.assertEqual([name for name, _ in registry.source_items()], ["mic"])

        with self.assertRaises(ValueError):
            registry.add_source(source, running=False)

        with self.assertRaises(RuntimeError):
            registry.add_source(_FakeSource(name="other"), running=True)

    def test_engine_runtime_state_owns_counters_tap_and_autosync(self) -> None:
        runtime = EngineRuntimeState(tap_queue_max=25)
        tap_q: queue.Queue[dict] = queue.Queue()

        runtime.set_tap_queue(tap_q)
        runtime.apply_tap_config(normalize_tap_config(mode="mix", drop_threshold=0.2))
        runtime.enable_auto_sync("speaker", "mic")
        runtime.record_master_metrics(0.5, 12.0)
        runtime.record_output_drop()
        runtime.record_tap_drop()
        t_start, t_end = runtime.next_mix_window(0.1)

        self.assertIs(runtime.tap_q, tap_q)
        self.assertEqual(runtime.tap_queue_max, 25)
        self.assertEqual(runtime.tap_mode, "mix")
        self.assertEqual(runtime.tap_drop_threshold, 0.2)
        self.assertEqual((t_start, t_end), (0.0, 0.1))
        self.assertEqual(runtime.tick_index, 1)
        self.assertEqual(runtime.master_rms, 0.5)
        self.assertEqual(runtime.master_last_ts, 12.0)
        self.assertEqual(runtime.dropped_out_blocks, 1)
        self.assertEqual(runtime.dropped_tap_blocks, 1)
        self.assertTrue(runtime.autosync_enabled)
        self.assertEqual(runtime.autosync_ref, "speaker")
        self.assertEqual(runtime.autosync_target, "mic")

        runtime.disable_auto_sync()
        runtime.reset_after_stop()

        self.assertFalse(runtime.autosync_enabled)
        self.assertEqual(runtime.master_rms, 0.0)
        self.assertEqual(runtime.master_last_ts, 0.0)
        self.assertEqual(runtime.dropped_out_blocks, 0)
        self.assertEqual(runtime.dropped_tap_blocks, 0)
