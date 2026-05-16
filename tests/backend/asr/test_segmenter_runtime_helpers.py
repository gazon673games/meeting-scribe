from __future__ import annotations

import queue
import unittest

import numpy as np

from asr.application.metrics import ASRMetrics
from asr.application.segmentation import SegmenterConfig, StreamingSegmenterConfig
from asr.domain.segments import Segment
from asr.infrastructure import segmentation as batch_segmentation
from asr.infrastructure import streaming_segmenter
from asr.infrastructure.audio_data import MonoAudio16kBuffer


class _FakeVAD:
    frame_len = 320
    frame_ms = 20

    def __init__(self, speech: list[bool] | None = None) -> None:
        self._speech = list(speech or [])
        self.reset_called = False

    def is_speech_frame(self, frame) -> bool:  # noqa: ANN001
        return self._speech.pop(0) if self._speech else False

    def pop_preroll(self):
        return np.arange(640, dtype=np.float32), None

    def speech_long_enough(self) -> bool:
        return True

    def reset(self) -> None:
        self.reset_called = True

    def last_rms(self) -> float:
        return 0.1

    def last_threshold(self) -> float:
        return 0.2

    def noise_rms(self) -> float:
        return 0.01

    def last_band_ratio(self) -> float:
        return 0.3

    def last_voiced(self) -> float:
        return 0.4


class _Diarization:
    def __init__(self) -> None:
        self.streams = []
        self.ring_updates = []

    def ensure_stream(self, name: str) -> None:
        self.streams.append(name)

    def update_ring(self, stream: str, t_end: float, audio) -> None:  # noqa: ANN001
        self.ring_updates.append((stream, t_end, audio.shape[0]))


class SegmenterRuntimeHelperTests(unittest.TestCase):
    def _config(self) -> SegmenterConfig:
        return SegmenterConfig(
            vad_energy_threshold=0.01,
            vad_hangover_ms=20,
            vad_min_speech_ms=20,
            vad_band_ratio_min=0.1,
            vad_voiced_min=0.1,
            vad_pre_speech_ms=20,
            vad_min_end_silence_ms=0,
            min_segment_ms=1,
            agc_enabled=False,
            agc_target_rms=0.1,
            agc_max_gain=2.0,
            agc_alpha=1.0,
        )

    def test_batch_audio_segmenter_feeds_packets_finalizes_and_enqueues_segments(self) -> None:
        seg_q: queue.Queue = queue.Queue(maxsize=2)
        diar_q: queue.Queue = queue.Queue(maxsize=1)
        logs = []
        diarization = _Diarization()
        segmenter = batch_segmentation.AudioSegmenter(
            config=self._config(),
            segment_queue=seg_q,
            diarization_queue=diar_q,
            diarization=diarization,
            metrics=ASRMetrics(latency_window=20, emit_interval_s=1.0),
            log_event=logs.append,
            segmentation_params=lambda: (20.0, 1.0, 20.0),
        )

        segmenter.reset_runtime()
        segmenter.feed_packet(mode="split", pkt={"t_start": 0.0, "t_end": 0.01, "sources": {"mic": np.zeros(10)}})
        segmenter.feed_packet(mode="mix", pkt={"t_start": 0.0, "t_end": 0.01, "mix": np.zeros(10)})
        segmenter.feed_stream("mic", 0.0, 0.01, np.zeros(10, dtype=np.float32), sample_rate=16000)

        state = batch_segmentation._StreamState(
            vad=_FakeVAD([True, False]),
            buffer=[],
            speech_start_ts=None,
            residual=np.zeros(0, dtype=np.float32),
            agc=None,
        )
        segmenter._streams["mic"] = state
        segmenter._run_vad_loop("mic", state, 0.0, 0.04, np.ones(640, dtype=np.float32))
        segmenter._heartbeat("mic", state)

        self.assertFalse(seg_q.empty())
        self.assertFalse(diar_q.empty())
        self.assertTrue(any(event.get("type") == "segment_ready" for event in logs))
        self.assertEqual(diarization.streams[0], "mic")

        if diar_q.empty():
            diar_q.put_nowait(Segment("mic", 0.0, 0.1, MonoAudio16kBuffer.from_array([0.0]), 0.0))
        segmenter._enqueue_diarization_segment(Segment("mic", 0.0, 0.1, MonoAudio16kBuffer.from_array([0.0]), 0.0))
        self.assertTrue(any(event.get("type") == "diar_segment_dropped" for event in logs))

    def test_batch_audio_segmenter_handles_invalid_and_full_queue_paths(self) -> None:
        seg_q: queue.Queue = queue.Queue(maxsize=1)
        logs = []
        segmenter = batch_segmentation.AudioSegmenter(
            config=self._config(),
            segment_queue=seg_q,
            diarization_queue=None,
            diarization=_Diarization(),
            metrics=ASRMetrics(latency_window=20, emit_interval_s=1.0),
            log_event=logs.append,
            segmentation_params=lambda: (20.0, 1.0, 0.0),
        )
        state = batch_segmentation._StreamState(
            vad=_FakeVAD(),
            buffer=[np.zeros(320, dtype=np.float32)],
            speech_start_ts=0.0,
            residual=np.zeros(0, dtype=np.float32),
            agc=None,
        )
        segmenter._streams["mic"] = state
        self.assertTrue(segmenter._is_segment_valid(state, np.ones(320, dtype=np.float32)))
        segmenter._finalize_segment("mic", state, 0.1)

        if seg_q.empty():
            seg_q.put_nowait(Segment("mic", 0.0, 0.1, MonoAudio16kBuffer.from_array([0.0]), 0.0))
        segmenter._enqueue_segment("mic", 0.0, 0.1, np.ones(320, dtype=np.float32))
        self.assertTrue(any(event.get("type") == "segment_dropped" for event in logs))

    def test_streaming_audio_segmenter_feeds_packets_and_emits_chunks(self) -> None:
        chunk_q: queue.Queue = queue.Queue(maxsize=2)
        logs = []
        segmenter = streaming_segmenter.StreamingAudioSegmenter(
            config=StreamingSegmenterConfig(
                chunk_interval_s=0.0,
                endpoint_silence_ms=20.0,
                max_segment_s=1.0,
                agc_enabled=False,
            ),
            chunk_queue=chunk_q,
            log_event=logs.append,
        )

        segmenter.reset_runtime()
        segmenter.feed_packet(mode="mix", pkt={"t_start": 0.0, "t_end": 0.01, "mix": np.zeros(10)})
        segmenter.feed_packet(mode="split", pkt={"t_start": 0.0, "t_end": 0.01, "sources": {"mic": np.zeros(10)}})
        segmenter.feed_stream("mic", 0.0, 0.01, np.zeros(10, dtype=np.float32), sample_rate=16000)

        state = streaming_segmenter._StreamState(
            vad=_FakeVAD([True, False]),
            buffer=[],
            speech_start_ts=None,
            last_chunk_ts=0.0,
            residual=np.zeros(0, dtype=np.float32),
            agc=None,
        )
        segmenter._streams["mic"] = state
        streaming_segmenter._prepend_preroll(state)
        segmenter._run_vad_loop("mic", state, 0.0, 0.04, np.ones(640, dtype=np.float32))

        self.assertFalse(chunk_q.empty())

        full_q: queue.Queue = queue.Queue(maxsize=1)
        full_q.put(object())
        segmenter_full = streaming_segmenter.StreamingAudioSegmenter(
            config=StreamingSegmenterConfig(agc_enabled=False),
            chunk_queue=full_q,
            log_event=logs.append,
        )
        full_state = streaming_segmenter._StreamState(
            vad=_FakeVAD(),
            buffer=[np.ones(320, dtype=np.float32)],
            speech_start_ts=0.0,
            last_chunk_ts=0.0,
            residual=np.zeros(0, dtype=np.float32),
            agc=None,
        )
        segmenter_full._emit_chunk("mic", full_state, 0.1, is_final=False)
        self.assertTrue(any(event.get("type") == "streaming_chunk_dropped" for event in logs))


if __name__ == "__main__":
    unittest.main()
