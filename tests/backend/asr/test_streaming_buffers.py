from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from asr.application.segmentation import StreamingSegmenterConfig
from asr.infrastructure.audio_data import ChunkedAudio16kBuffer
from asr.infrastructure.streaming_queue import CoalescingStreamingQueue
from asr.infrastructure.streaming_segmenter import StreamingAudioSegmenter


class StreamingBufferTests(unittest.TestCase):
    def test_emission_and_coalescing_leave_full_audio_unmaterialized(self):
        pending = CoalescingStreamingQueue()
        segmenter = StreamingAudioSegmenter(
            config=StreamingSegmenterConfig(),
            chunk_queue=pending,
            log_event=lambda event: None,
        )
        state = segmenter._ensure_stream('mic')
        state.speech_start_ts = 0
        first = np.ones(320, np.float32)
        second = np.full(320, 2, np.float32)
        state.buffer.append(first)
        original_concatenate = np.concatenate
        with patch(
            'asr.infrastructure.audio_data.np.concatenate',
            wraps=original_concatenate,
        ) as concatenate:
            segmenter._emit_chunk('mic', state, 0.02, is_final=False)
            state.buffer.append(second)
            segmenter._emit_chunk('mic', state, 0.04, is_final=True)
            concatenate.assert_not_called()
            final = pending.get(timeout=0)
            self.assertNotIn('samples', final.audio.__dict__)
            np.testing.assert_array_equal(
                final.incremental_audio.samples, np.r_[first, second]
            )
            self.assertNotIn('samples', final.audio.__dict__)
        self.assertTrue(final.is_final)
        self.assertTrue(pending.empty())

    def test_tail_handles_partial_frames_and_preserves_full_whisper_context(
        self,
    ):
        audio = ChunkedAudio16kBuffer(
            (np.arange(3, dtype=np.float32), np.arange(3, 7, dtype=np.float32))
        )
        np.testing.assert_array_equal(audio.tail(2).samples, [2, 3, 4, 5, 6])
        np.testing.assert_array_equal(audio.samples, np.arange(7))
        self.assertIs(audio.samples, audio.samples)
        self.assertEqual(audio.tail(7).frame_count, 0)
        self.assertEqual(audio.tail(100).frame_count, 0)

    def test_consumer_cannot_modify_the_queued_frames_through_samples(self):
        frame = np.arange(3, dtype=np.float32)
        audio = ChunkedAudio16kBuffer((frame,))
        audio.samples[0] = 99
        np.testing.assert_array_equal(frame, [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
