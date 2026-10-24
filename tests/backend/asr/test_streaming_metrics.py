from __future__ import annotations

import queue
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from application.event_types import event_from_record, event_to_record
from asr.application.metrics import ASRMetrics
from asr.application.streaming_worker_config import StreamingWorkerConfig
from asr.domain.streaming import StreamingChunk
from asr.infrastructure.audio_data import MonoAudio16kBuffer
from asr.infrastructure.streaming_worker import StreamingWhisperWorker


class StreamingMetricsTests(unittest.TestCase):
    def test_streaming_reports_cpu_inference_queue_wait_rtf_and_wire_roundtrip(
        self,
    ):
        records = []
        clock = Mock(return_value=10.0)
        backend = Mock(device='cpu')

        def transcribe(*args, **kwargs):
            clock.return_value += 0.2
            return {'text': 'hello'}

        backend.transcribe_stream_chunk.side_effect = transcribe
        worker = StreamingWhisperWorker(
            config=StreamingWorkerConfig(
                'native', 'en', 'cuda', 'int8', 1, 1, None
            ),
            chunk_queue=queue.Queue(),
            stop_event=threading.Event(),
            log_event=records.append,
            asr_backend_factory=Mock(),
        )
        worker._bind_backend(backend)
        audio = MonoAudio16kBuffer.from_array(np.zeros(16000, np.float32))
        chunk = StreamingChunk('mic', 0, 1, audio, True, 1000, audio, 9.5)
        with patch('asr.infrastructure.streaming_worker.time.monotonic', clock):
            worker._process(chunk)

        metrics = next(
            record for record in records if record['type'] == 'asr_metrics'
        )
        self.assertAlmostEqual(metrics['avg_latency_s'], 0.2)
        self.assertAlmostEqual(metrics['queue_wait_s'], 0.5)
        self.assertAlmostEqual(metrics['lag_s'], 0.7)
        self.assertAlmostEqual(metrics['rtf'], 0.2)
        self.assertEqual(metrics['device'], 'cpu')
        parsed = event_to_record(event_from_record(metrics))
        for field in ('queue_wait_s', 'rtf', 'seg_qsize', 'device'):
            self.assertEqual(parsed[field], metrics[field])

        backend.transcribe_stream_chunk.side_effect = RuntimeError(
            'decode failed'
        )
        records.clear()
        worker._process(chunk)
        self.assertFalse(
            any(record['type'] == 'asr_metrics' for record in records)
        )
        self.assertTrue(any(record['type'] == 'error' for record in records))

    def test_metric_cadence_ignores_wall_clock_jumps_and_p95_uses_upper_rank(
        self,
    ):
        metrics = ASRMetrics(latency_window=20, emit_interval_s=1)
        metrics.record_latency(asr_latency_s=1, total_lag_s=1)
        metrics.record_latency(asr_latency_s=2, total_lag_s=2)
        options = {
            'force': False,
            'seg_qsize': 0,
            'overload_active': False,
            'overload_strategy': 'coalesce',
            'hard_overload': False,
        }
        with (
            patch(
                'asr.application.metrics.time.monotonic',
                side_effect=[10, 10.5, 11],
            ),
            patch('asr.application.metrics.time.time', side_effect=[1000, 1]),
        ):
            first = metrics.build_event(**options)
            self.assertIsNone(metrics.build_event(**options))
            last = metrics.build_event(**options)
        self.assertEqual(first['p95_latency_s'], 2)
        self.assertEqual(last['ts'], 1)


if __name__ == '__main__':
    unittest.main()
