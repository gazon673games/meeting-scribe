from __future__ import annotations

import queue
import unittest
from unittest.mock import Mock

import numpy as np

from asr.application.metrics import ASRMetrics
from asr.application.pipeline_config import (
    ASRPipelineSettings,
    build_segmenter_config,
    build_streaming_segmenter_config,
)
from asr.infrastructure.segmentation import AudioSegmenter
from asr.infrastructure.streaming_segmenter import StreamingAudioSegmenter
from diarization.application.diarization import DiarizationPort


class SegmentationEndpointTests(unittest.TestCase):
    def test_silence_endpoint_accumulates_across_packets(self) -> None:
        settings = ASRPipelineSettings(
            endpoint_silence_ms=60.0,
            streaming_endpoint_silence_ms=60.0,
            max_segment_s=10.0,
            min_segment_ms=20,
            vad_hangover_ms=0,
            vad_min_speech_ms=20,
            vad_pre_speech_ms=0,
            vad_min_end_silence_ms=0,
            agc_enabled=False,
        )
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                output = queue.Queue()
                if streaming:
                    segmenter = StreamingAudioSegmenter(
                        config=build_streaming_segmenter_config(settings),
                        chunk_queue=output,
                        log_event=lambda event: None,
                    )
                else:
                    segmenter = AudioSegmenter(
                        config=build_segmenter_config(settings),
                        segment_queue=output,
                        diarization=Mock(spec=DiarizationPort),
                        metrics=ASRMetrics.from_settings(settings),
                        log_event=lambda event: None,
                        segmentation_params=lambda: (60.0, 10.0, 0.0),
                    )
                times = np.arange(320, dtype=np.float64) / 16000.0
                speech = (0.2 * np.sin(2 * np.pi * 1000 * times)).astype(
                    np.float32
                )
                segmenter.feed_stream(
                    'mic', 0.0, 0.02, speech, sample_rate=16000
                )
                for index in range(1, 7):
                    segmenter.feed_stream(
                        'mic',
                        index * 0.02,
                        (index + 1) * 0.02,
                        np.zeros(320, dtype=np.float32),
                        sample_rate=16000,
                    )

                self.assertEqual(output.qsize(), 1)
                segment = output.get_nowait()
                self.assertEqual(segment.stream, 'mic')
                self.assertLess(segment.t_end, settings.max_segment_s)
                if streaming:
                    self.assertTrue(segment.is_final)


if __name__ == '__main__':
    unittest.main()
