from __future__ import annotations

import unittest

import numpy as np

from asr.application.metrics import ASRMetrics
from asr.application.overload import OverloadController
from asr.application.pipeline_config import (
    ASRPipelineSettings,
    build_segmenter_config,
    build_streaming_segmenter_config,
)
from asr.application.policies import AdaptiveBeam
from asr.application.streaming_worker_config import StreamingWorkerConfig
from asr.application.utterances import UtteranceAggregator
from asr.application.worker_config import TranscriptionWorkerConfig
from asr.domain.dedup import StreamDedupFilter
from asr.domain.segments import Segment
from asr.domain.streaming import ConfirmedPrefixTracker, StreamingWord
from asr.domain.text import normalize_text, trim_overlap
from asr.domain.utterances import UtteranceState


class ASRConfigAndDomainHelperTests(unittest.TestCase):
    def test_pipeline_configs_worker_configs_and_event_dicts_map_settings(self) -> None:
        settings = ASRPipelineSettings(
            asr_model_name="tiny",
            asr_language="en",
            asr_initial_prompt="terms",
            source_speaker_labels={"mic": "Alice"},
            diarization_enabled=True,
            diarization_sidecar_enabled=False,
            overload_strategy="keep_all",
            adaptive_beam_max=None,
            beam_size=7,
            streaming_chunk_interval_s=0.25,
        )

        segmenter = build_segmenter_config(settings)
        streaming_segmenter = build_streaming_segmenter_config(settings)
        worker = TranscriptionWorkerConfig.from_settings(settings)
        streaming_worker = StreamingWorkerConfig.from_settings(settings)

        self.assertEqual(settings.normalized_overload_strategy, "keep_all")
        self.assertEqual(settings.resolved_adaptive_beam_max, 7)
        self.assertEqual(segmenter.to_event_dict()["energy_threshold"], settings.vad_energy_threshold)
        self.assertEqual(streaming_segmenter.chunk_interval_s, 0.25)
        self.assertEqual(worker.source_speaker_labels, {"mic": "Alice"})
        self.assertTrue(worker.diarization_blocking_lookup)
        self.assertEqual(streaming_worker.model_name, "tiny")

    def test_metrics_overload_and_adaptive_beam_emit_state_transitions(self) -> None:
        settings = ASRPipelineSettings(metrics_latency_window=2, metrics_emit_interval_s=0.25, overload_strategy="keep_all")
        metrics = ASRMetrics.from_settings(settings)
        metrics.record_segment_dropped(2)
        metrics.record_segments_skipped(3)
        metrics.record_latency(asr_latency_s=0.4, total_lag_s=0.8)

        event = metrics.build_event(
            force=True,
            seg_qsize=5,
            overload_active=True,
            overload_strategy="keep_all",
            hard_overload=False,
        )

        self.assertEqual(event["seg_dropped_total"], 2)
        self.assertEqual(event["seg_skipped_total"], 3)
        self.assertIsNone(metrics.build_event(force=False, seg_qsize=5, overload_active=True, overload_strategy="keep_all", hard_overload=False))
        metrics.reset()
        self.assertEqual(metrics.seg_dropped_total, 0)

        controller = OverloadController.from_settings(
            ASRPipelineSettings(
                overload_strategy="keep_all",
                overload_enter_qsize=3,
                overload_exit_qsize=1,
                overload_hard_drop_qsize=4,
                overload_hold_s=0.1,
                overload_beam_cap=2,
            )
        )
        enter_events = controller.update(seg_qsize=4, beam_cur=5, lag_s=1.2, now=10.0)
        self.assertEqual(enter_events[0]["type"], "asr_overload")
        self.assertEqual(controller.limit_beam(5), 1)
        self.assertEqual(controller.drop_old_count(8), 0)
        self.assertLessEqual(controller.segmentation_params(endpoint_silence_ms=800, max_segment_s=12, overlap_ms=300)[1], 5.0)
        self.assertEqual(controller.to_event_dict()["beam_cap"], 2)
        exit_events = controller.update(seg_qsize=1, beam_cur=1, lag_s=0.1, now=11.0)
        self.assertTrue(exit_events)
        controller.reset()
        self.assertFalse(controller.active)

        beam = AdaptiveBeam.from_settings(ASRPipelineSettings(beam_size=4, adaptive_beam_min=1, adaptive_beam_max=4))
        self.assertEqual(beam.maybe_update(seg_qsize=12, last_latency_s=2.0, last_dur_s=1.0, now=10.0)[0], 3)
        self.assertEqual(beam.maybe_update(seg_qsize=0, last_latency_s=0.1, last_dur_s=1.0, now=13.0)[0], 4)

    def test_streaming_dedup_text_segment_and_utterance_domain_helpers_behave(self) -> None:
        self.assertEqual(normalize_text(" a   b "), "a b")
        self.assertEqual(trim_overlap("hello world", "world again", max_window=20, min_match=5), ("again", 5))

        dedup = StreamDedupFilter(enabled=True, window=40, min_match=5)
        self.assertEqual(dedup.filter("mic", "hello world"), ("hello world", 0))
        self.assertEqual(dedup.filter("mic", "world again"), ("again", 5))
        dedup.reset()
        self.assertEqual(StreamDedupFilter(enabled=False, window=10).filter("mic", "same"), ("same", 0))

        words = [
            StreamingWord("one", 0.0, 0.1),
            StreamingWord("two", 0.1, 0.2),
            StreamingWord("three", 0.2, 0.3),
        ]
        tracker = ConfirmedPrefixTracker(lookahead=1)
        self.assertEqual(tracker.update(words).newly_confirmed, [])
        self.assertEqual([word.text for word in tracker.update(words).newly_confirmed], ["one", "two"])
        tracker.flush()
        self.assertEqual([word.text for word in tracker.confirmed_words], ["one", "two", "three"])
        tracker.reset()
        self.assertEqual(tracker.confirmed_words, [])

        segment = Segment("mic", 1.0, 2.5, np.zeros(4, dtype=np.float32), enqueue_ts=1.25)
        self.assertEqual(segment.queue_wait_s(2.0), 0.75)

        utterance = UtteranceState("mic", "Alice", 0.0, 1.0, "hello", last_emit_ts=10.0)
        self.assertEqual(utterance.duration_s, 1.0)
        self.assertTrue(utterance.can_extend(t_start=1.2, t_end=2.0, gap_s=0.5, max_s=3.0))
        utterance.extend(t_end=2.0, text="world", last_emit_ts=11.0)
        self.assertEqual(utterance.text, "hello world")
        self.assertTrue(utterance.should_flush(now=14.0, flush_s=2.0, force=False))

    def test_utterance_aggregator_flushes_grouped_text(self) -> None:
        aggregator = UtteranceAggregator.from_settings(
            ASRPipelineSettings(utterance_gap_s=0.5, utterance_max_s=3.0, utterance_flush_s=1.0)
        )

        self.assertEqual(
            aggregator.update(
                stream="mic",
                speaker="Alice",
                text="hello",
                t_start=0.0,
                t_end=0.5,
                now=1.0,
                overload=False,
            ),
            [],
        )
        self.assertEqual(
            aggregator.update(
                stream="mic",
                speaker="Alice",
                text="world",
                t_start=0.7,
                t_end=1.0,
                now=1.2,
                overload=False,
            ),
            [],
        )
        emitted = aggregator.flush_all(now=3.0, force=True, overload=False)

        self.assertEqual(emitted[0]["text"], "hello world")
        self.assertEqual(aggregator.to_event_dict()["utterance_enabled"], True)
        self.assertEqual(
            UtteranceAggregator(enabled=False, gap_s=0.5, max_s=3.0, flush_s=1.0, log_speaker_labels=True).update(
                stream="mic",
                speaker="Alice",
                text="ignored",
                t_start=0.0,
                t_end=0.5,
                now=1.0,
                overload=False,
            ),
            [],
        )
        self.assertEqual(
            aggregator.update(
                stream="mic",
                speaker="Alice",
                text="first",
                t_start=5.0,
                t_end=5.2,
                now=5.0,
                overload=False,
            ),
            [],
        )
        flushed = aggregator.update(
            stream="mic",
            speaker="Alice",
            text="second",
            t_start=10.0,
            t_end=10.2,
            now=10.0,
            overload=True,
        )
        self.assertEqual(flushed[0]["text"], "first")


if __name__ == "__main__":
    unittest.main()
