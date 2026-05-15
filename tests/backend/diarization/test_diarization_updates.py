from __future__ import annotations

import queue
import unittest

import numpy as np

from asr.domain.segments import Segment
from asr.infrastructure.audio_data import MonoAudio16kBuffer
from diarization.application.diarization_updates import DiarizationUpdateConfig, DiarizationUpdateRuntime


class _Stop:
    def is_set(self) -> bool:
        return False


class _Diarization:
    enabled = True
    backend = "online"

    def __init__(self, speaker: str) -> None:
        self.speaker = speaker
        self.started = False

    def ensure_stream(self, name: str) -> None:
        pass

    def update_ring(self, stream: str, t1: float, audio_16k) -> None:  # noqa: ANN001
        pass

    def init_backend(self, log_event) -> None:  # noqa: ANN001
        self.started = True

    def speaker_for_segment(self, seg, log_event) -> str:  # noqa: ANN001
        return self.speaker


def _segment() -> Segment:
    return Segment(
        stream="desktop_audio",
        t_start=1.0,
        t_end=2.0,
        audio=MonoAudio16kBuffer.from_array(np.zeros(16000, dtype=np.float32)),
        enqueue_ts=0.0,
    )


class DiarizationUpdateRuntimeTests(unittest.TestCase):
    def test_emits_speaker_update_for_non_fallback_speaker(self) -> None:
        segment_queue: queue.Queue[Segment] = queue.Queue()
        segment_queue.put_nowait(_segment())
        records: list[dict] = []

        runtime = DiarizationUpdateRuntime(
            config=DiarizationUpdateConfig(
                enabled=True,
                source_speaker_labels={"desktop_audio": "Remote"},
            ),
            segment_queue=segment_queue,
            stop_event=_Stop(),
            diarization=_Diarization("Remote S1"),
            log_event=records.append,
        )

        self.assertTrue(runtime.run_once())

        updates = [record for record in records if record.get("type") == "transcript_speaker_update"]
        self.assertEqual(updates[0]["speaker"], "Remote S1")
        self.assertEqual(updates[0]["stream"], "desktop_audio")

    def test_skips_source_fallback_speaker(self) -> None:
        segment_queue: queue.Queue[Segment] = queue.Queue()
        segment_queue.put_nowait(_segment())
        records: list[dict] = []

        runtime = DiarizationUpdateRuntime(
            config=DiarizationUpdateConfig(
                enabled=True,
                source_speaker_labels={"desktop_audio": "Remote"},
            ),
            segment_queue=segment_queue,
            stop_event=_Stop(),
            diarization=_Diarization("Remote"),
            log_event=records.append,
        )

        self.assertTrue(runtime.run_once())

        self.assertFalse(any(record.get("type") == "transcript_speaker_update" for record in records))


if __name__ == "__main__":
    unittest.main()
