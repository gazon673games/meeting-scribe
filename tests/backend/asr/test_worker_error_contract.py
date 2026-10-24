from __future__ import annotations

import queue
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from asr.application.metrics import ASRMetrics
from asr.application.overload import OverloadController
from asr.application.pipeline_config import ASRPipelineSettings
from asr.application.policies import AdaptiveBeam
from asr.application.transcription_worker import TranscriptionWorkerRuntime
from asr.application.utterances import UtteranceAggregator
from asr.application.worker_config import TranscriptionWorkerConfig
from asr.domain.segments import Segment
from asr.infrastructure.audio_data import MonoAudio16kBuffer


class _Backend:
    def __init__(self, outcome, stop: threading.Event) -> None:
        self.outcome = outcome
        self.stop = stop
        self.closed = False

    def transcribe(self, samples, *, beam_size=None):
        self.stop.set()
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def close(self) -> None:
        self.closed = True


class WorkerErrorContractTests(unittest.TestCase):
    def test_only_operational_errors_are_recovered_per_segment(self) -> None:
        cases = [
            (RuntimeError('temporary model failure'), 'segment'),
            (TypeError('backend programming error'), 'worker'),
            ({'text': object()}, 'worker'),
        ]
        for outcome, expected_boundary in cases:
            with self.subTest(boundary=expected_boundary):
                settings = ASRPipelineSettings(diarization_enabled=False)
                stop = threading.Event()
                backend = _Backend(outcome, stop)
                segments = queue.Queue()
                segments.put(
                    Segment(
                        'mic',
                        0.0,
                        0.1,
                        MonoAudio16kBuffer.from_array(
                            np.zeros(1600, dtype=np.float32)
                        ),
                        enqueue_ts=0.0,
                    )
                )
                events = []
                worker = TranscriptionWorkerRuntime(
                    config=TranscriptionWorkerConfig.from_settings(settings),
                    segment_queue=segments,
                    stop_event=stop,
                    log_event=events.append,
                    metrics=ASRMetrics.from_settings(settings),
                    overload=OverloadController.from_settings(settings),
                    beam_controller=AdaptiveBeam.from_settings(settings),
                    diarization=SimpleNamespace(enabled=False, backend='off'),
                    utterances=UtteranceAggregator.from_settings(settings),
                    asr_backend_factory=lambda **options: backend,
                )
                worker.run_safe()
                failures = [event for event in events if 'error' in event]
                self.assertEqual(len(failures), 1)
                boundary = failures[0].get('where', failures[0]['type'])
                self.assertEqual(boundary, expected_boundary)
                self.assertTrue(backend.closed)
