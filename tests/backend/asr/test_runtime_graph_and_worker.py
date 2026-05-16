from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from asr.application.ingest import TapIngestRuntime
from asr.application.metrics import ASRMetrics
from asr.application.overload import OverloadController
from asr.application.pipeline import ASRPipeline
from asr.application.pipeline_config import ASRPipelineDependencies, ASRPipelineSettings
from asr.application.policies import AdaptiveBeam
from asr.application.transcription_worker import TranscriptionWorkerRuntime
from asr.application.utterances import UtteranceAggregator
from asr.application.worker_config import TranscriptionWorkerConfig
from asr.domain.segments import Segment
from asr.infrastructure.audio_data import MonoAudio16kBuffer


class _Handle:
    def __init__(self, name: str) -> None:
        self.name = name
        self.joined = False

    def join(self, timeout=None) -> None:  # noqa: ANN001
        self.joined = True


class _Runner:
    def __init__(self) -> None:
        self.started = []

    def create_stop_signal(self):
        return threading.Event()

    def start_worker(self, *, name: str, target):
        handle = _Handle(name)
        self.started.append((name, target, handle))
        return handle


class _Logger:
    max_bytes = 123
    backup_count = 2

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.records = []
        self.closed = False

    def write(self, rec: dict) -> None:
        self.records.append(rec)

    def close(self) -> None:
        self.closed = True


class _Segmenter:
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.kwargs = kwargs
        self.segmentation_params = kwargs.get("segmentation_params", lambda: None)()
        self.reset = False
        self.packets = []

    def reset_runtime(self) -> None:
        self.reset = True

    def feed_packet(self, *, mode: str, pkt: dict) -> None:
        self.packets.append((mode, pkt))


class _Diarization:
    enabled = True
    backend = "fake"

    def __init__(self) -> None:
        self.init_called = False

    def init_backend(self, log_event) -> None:  # noqa: ANN001
        self.init_called = True

    def speaker_for_segment(self, seg, log_event):  # noqa: ANN001
        return "D1"

    def identity_snapshot(self) -> dict:
        return {"speakers": ["D1"]}


class _ASRBackend:
    def __init__(self) -> None:
        self.closed = False

    def transcribe(self, samples, *, beam_size=None):  # noqa: ANN001
        return {"text": " hello world "}

    def close(self) -> None:
        self.closed = True


class ASRRuntimeGraphAndWorkerTests(unittest.TestCase):
    def _dependencies(self, runner: _Runner) -> ASRPipelineDependencies:
        return ASRPipelineDependencies(
            logger_factory=lambda **kwargs: _Logger(**kwargs),
            asr_backend_factory=lambda **kwargs: _ASRBackend(),
            worker_runner=runner,
            diarization_factory=lambda **kwargs: _Diarization(),
            segmenter_factory=lambda **kwargs: _Segmenter(**kwargs),
        )

    def test_pipeline_graph_start_stop_and_identity_snapshot_use_fake_runtime_parts(self) -> None:
        runner = _Runner()
        settings = ASRPipelineSettings(
            streaming_enabled=False,
            diarization_enabled=True,
            diarization_sidecar_enabled=True,
            diarization_queue_size=2,
        )
        events: queue.Queue = queue.Queue()

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = ASRPipeline(
                tap_queue=queue.Queue(),
                project_root=Path(tmp),
                settings=settings,
                dependencies=self._dependencies(runner),
                event_queue=events,
            )
            pipeline.start()
            params = pipeline.graph.segmentation_params(settings)
            pipeline.stop()

        self.assertEqual(params, (settings.endpoint_silence_ms, settings.max_segment_s, settings.overlap_ms))
        self.assertEqual(pipeline.identity_snapshot(), {"speakers": ["D1"]})
        self.assertEqual([item[0] for item in runner.started], ["asr-ingest", "asr-worker", "asr-diarization"])
        self.assertTrue(pipeline.graph.logger.closed)
        self.assertGreaterEqual(events.qsize(), 1)

    def test_streaming_pipeline_starts_streaming_worker_without_batch_worker(self) -> None:
        runner = _Runner()
        settings = ASRPipelineSettings(streaming_enabled=True)

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = ASRPipeline(
                tap_queue=queue.Queue(),
                project_root=Path(tmp),
                settings=settings,
                dependencies=self._dependencies(runner),
            )
            pipeline.start()
            pipeline.stop()

        self.assertEqual([item[0] for item in runner.started], ["asr-ingest", "asr-streaming"])
        self.assertIsNone(pipeline.graph.worker)
        self.assertIsNotNone(pipeline.graph.streaming_worker_runtime)

    def test_ingest_runtime_feeds_packets_and_reports_safe_errors(self) -> None:
        tap_queue: queue.Queue = queue.Queue()
        stop = threading.Event()
        segmenter = _Segmenter()
        logs = []

        tap_queue.put({"mix": np.zeros(4, dtype=np.float32)})
        original_feed_packet = segmenter.feed_packet

        def feed_once(*, mode: str, pkt: dict) -> None:
            original_feed_packet(mode=mode, pkt=pkt)
            stop.set()

        segmenter.feed_packet = feed_once  # type: ignore[method-assign]
        TapIngestRuntime(tap_queue=tap_queue, stop_event=stop, mode="mix", segmenter=segmenter, log_event=logs.append).run()
        self.assertEqual(segmenter.packets[0][0], "mix")

        class _BadSegmenter:
            def feed_packet(self, *, mode: str, pkt: dict) -> None:
                raise RuntimeError("feed failed")

        stop = threading.Event()
        tap_queue = queue.Queue()
        tap_queue.put({})
        runtime = TapIngestRuntime(tap_queue=tap_queue, stop_event=stop, mode="mix", segmenter=_BadSegmenter(), log_event=logs.append)
        runtime.run_safe()
        self.assertEqual(logs[-1]["where"], "ingest")

    def test_transcription_worker_processes_segments_metrics_overload_and_cleanup(self) -> None:
        seg_queue: queue.Queue = queue.Queue()
        logs = []
        stop = threading.Event()
        metrics = ASRMetrics(latency_window=20, emit_interval_s=0.25)
        overload = OverloadController(
            enter_qsize=1,
            exit_qsize=0,
            hard_qsize=2,
            hold_s=0.0,
            beam_cap=1,
            overlap_ms=80,
            max_segment_s=3.0,
            strategy="drop_old",
        )
        beam = AdaptiveBeam(min_beam=1, max_beam=4, cur_beam=4, last_change_ts=-10.0)
        diarization = _Diarization()
        utterances = UtteranceAggregator(enabled=True, gap_s=0.5, max_s=3.0, flush_s=0.0, log_speaker_labels=True)
        worker = TranscriptionWorkerRuntime(
            config=TranscriptionWorkerConfig(
                model_name="tiny",
                language="en",
                device="cpu",
                compute_type="int8",
                cpu_threads=1,
                num_workers=1,
                beam_size=4,
                initial_prompt=None,
                text_dedup_enabled=True,
                text_dedup_window=40,
                adaptive_beam_enabled=True,
                log_speaker_labels=True,
                init_diarization=True,
                diarization_blocking_lookup=True,
            ),
            segment_queue=seg_queue,
            stop_event=stop,
            log_event=logs.append,
            metrics=metrics,
            overload=overload,
            beam_controller=beam,
            diarization=diarization,
            utterances=utterances,
            asr_backend_factory=lambda **kwargs: _ASRBackend(),
        )
        segment = Segment("mic", 0.0, 1.0, MonoAudio16kBuffer.from_array(np.ones(1600, dtype=np.float32)), enqueue_ts=0.0)

        worker.reset_runtime()
        worker._asr = _ASRBackend()
        worker._transcribe_segment(segment)
        worker.flush_utterances(force=True)
        worker.emit_metrics(force=True)

        seg_queue.put(segment)
        seg_queue.put(segment)
        worker._drain_old_segments_if_hard_overload()
        worker._update_overload_state()

        stop.set()
        worker.run()
        worker.run_safe()

        self.assertTrue(any(event.get("type") == "segment" for event in logs))
        self.assertTrue(any(event.get("type") == "asr_metrics" for event in logs))
        self.assertTrue(diarization.init_called)


if __name__ == "__main__":
    unittest.main()
