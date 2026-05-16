from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from interface.session_controller import HeadlessSessionController
from tests.helpers.electron_interface_fakes import (
    _FakeAudioRuntimeFactory,
    _FakeAudioSourceFactory,
    _FakeWavRecorderFactory,
)


class _TranscriptStore:
    def __init__(self) -> None:
        self.current_human_log_path = Path("human.log")
        self.realtime_transcript_path = Path("live.srt")
        self.realtime_enabled = False
        self.human_opened = False
        self.closed = []
        self.human_lines = []
        self.srt_entries = []

    def set_realtime_enabled(self, enabled: bool) -> None:
        self.realtime_enabled = bool(enabled)

    def open_human_log(self) -> Path:
        self.human_opened = True
        return self.current_human_log_path

    def close_human_log(self) -> None:
        self.closed.append("human")

    def close_realtime_transcript(self) -> None:
        self.closed.append("realtime")

    def write_human_line(self, line: str) -> None:
        self.human_lines.append(line)

    def write_realtime_srt_entry(self, ts, stream, text, *, speaker):  # noqa: ANN001
        self.srt_entries.append({"ts": ts, "stream": stream, "text": text, "speaker": speaker})


def _controller(root: Path, *, transcript_store=None, event_sink=None):  # noqa: ANN001
    return HeadlessSessionController(
        project_root=root,
        audio_runtime_factory=_FakeAudioRuntimeFactory(),
        audio_source_factory=_FakeAudioSourceFactory(),
        wav_recorder_factory=_FakeWavRecorderFactory(),
        transcript_store=transcript_store,
        event_sink=event_sink,
    )


class TranscriptPipelineTests(unittest.TestCase):
    def test_transcript_store_lifecycle_writes_human_and_realtime_lines(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = _TranscriptStore()
            controller = _controller(Path(raw_root), transcript_store=store)
            controller._asr_running = True  # noqa: SLF001

            controller._configure_transcript_files_locked({"realtimeTranscriptToFile": True})  # noqa: SLF001
            controller._write_transcript_line_locked(  # noqa: SLF001
                {"text": "hello", "ts": 10.0, "stream": "mic", "speaker": "Speaker 1"}
            )
            controller._close_transcript_files_locked()  # noqa: SLF001

            self.assertTrue(store.realtime_enabled)
            self.assertTrue(store.human_opened)
            self.assertEqual(controller.snapshot()["humanLogPath"], "human.log")
            self.assertIn("Speaker 1: hello", store.human_lines[0])
            self.assertEqual(store.srt_entries, [{"ts": 10.0, "stream": "mic", "text": "hello", "speaker": "Speaker 1"}])
            self.assertEqual(store.closed, ["human", "realtime"])
            self.assertFalse(controller._realtime_transcript_enabled)  # noqa: SLF001

    def test_streaming_words_final_metrics_and_source_errors_update_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            events: list[tuple[str, dict]] = []
            controller = _controller(Path(raw_root), event_sink=lambda kind, payload: events.append((kind, payload)))

            controller._handle_asr_event(  # noqa: SLF001
                {
                    "type": "streaming_words",
                    "stream": "mic",
                    "confirmed": [{"text": "hello"}],
                    "tentative": [{"text": "there"}],
                    "t_start": "1.0",
                    "t_end": "1.5",
                    "ts": 2.0,
                }
            )
            controller._handle_asr_event(  # noqa: SLF001
                {
                    "type": "streaming_words",
                    "stream": "mic",
                    "confirmed": [{"text": "friend"}],
                    "tentative": [{"text": "now"}],
                    "t_end": 2.0,
                }
            )
            self.assertEqual(controller.snapshot()["transcript"][0]["text"], "hello friend now")

            controller._handle_asr_event(  # noqa: SLF001
                {
                    "type": "streaming_final",
                    "stream": "mic",
                    "words": [{"text": "final"}, {"text": "text"}],
                    "t_start": 1.0,
                    "t_end": 2.2,
                    "ts": 3.0,
                }
            )
            controller._handle_asr_event(  # noqa: SLF001
                {
                    "type": "asr_metrics",
                    "seg_dropped_total": 1,
                    "seg_skipped_total": 2,
                    "avg_latency_s": 0.3,
                    "p95_latency_s": 0.8,
                    "lag_s": 0.1,
                }
            )
            controller._handle_asr_event({"type": "source_error", "source": "mic", "error": "device lost"})  # noqa: SLF001

            snapshot = controller.snapshot()
            self.assertEqual(len(snapshot["transcript"]), 1)
            self.assertEqual(snapshot["transcript"][0]["text"], "final text")
            self.assertEqual(snapshot["asrMetrics"]["segDroppedTotal"], 1)
            self.assertEqual(snapshot["lastError"], "mic: device lost")
            self.assertGreaterEqual(len([kind for kind, _ in events if kind == "asr_event"]), 4)


if __name__ == "__main__":
    unittest.main()
