from __future__ import annotations

import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from asr.application.streaming_worker_config import StreamingWorkerConfig
from asr.domain.streaming import StreamingChunk, StreamingWord
from asr.infrastructure.audio_data import MonoAudio16kBuffer
from asr.infrastructure.logger import ASRLogger, _EventFilter
from asr.infrastructure.streaming_worker import StreamingWhisperWorker, _word_to_dict
from asr.infrastructure.worker_faster_whisper import FasterWhisperASR


class _StreamingBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.closed = False

    def transcribe_words(self, samples):  # noqa: ANN001
        if self.fail:
            raise RuntimeError("decode failed")
        return [
            {"text": "hello", "start": 0.0, "end": 0.2},
            {"text": "world", "start": 0.2, "end": 0.4},
        ]

    def close(self) -> None:
        self.closed = True


class StreamingWorkerAndLoggerTests(unittest.TestCase):
    def _chunk(self, *, final: bool = False) -> StreamingChunk:
        return StreamingChunk(
            stream="mic",
            t_start=0.0,
            t_end=1.0,
            audio=MonoAudio16kBuffer.from_array(np.ones(320, dtype=np.float32)),
            is_final=final,
            enqueue_ts=0.0,
        )

    def test_streaming_worker_processes_intermediate_final_error_and_cleanup_paths(self) -> None:
        logs = []
        worker = StreamingWhisperWorker(
            config=StreamingWorkerConfig(
                model_name="tiny",
                language="en",
                device="cpu",
                compute_type="int8",
                cpu_threads=1,
                num_workers=1,
                initial_prompt=None,
                lookahead=1,
                queue_timeout_s=0.01,
            ),
            chunk_queue=queue.Queue(),
            stop_event=threading.Event(),
            log_event=logs.append,
            asr_backend_factory=lambda **kwargs: _StreamingBackend(),
        )

        worker._asr = _StreamingBackend()
        worker._process(self._chunk(final=False))
        worker._process(self._chunk(final=True))
        worker.reset_runtime()

        worker._asr = _StreamingBackend(fail=True)
        worker._process(self._chunk(final=False))

        stop = threading.Event()
        stop.set()
        worker = StreamingWhisperWorker(
            config=StreamingWorkerConfig(
                model_name="tiny",
                language=None,
                device="cpu",
                compute_type="int8",
                cpu_threads=1,
                num_workers=1,
                initial_prompt="terms",
                queue_timeout_s=0.01,
            ),
            chunk_queue=queue.Queue(),
            stop_event=stop,
            log_event=logs.append,
            asr_backend_factory=lambda **kwargs: _StreamingBackend(),
        )
        worker.run()
        worker.run_safe()

        self.assertTrue(any(event.get("type") == "streaming_words" for event in logs))
        self.assertTrue(any(event.get("type") == "streaming_final" for event in logs))
        self.assertTrue(any(event.get("where") == "streaming_transcribe" for event in logs))
        self.assertEqual(_word_to_dict(StreamingWord("x", 1.0, 2.0)), {"text": "x", "start": 1.0, "end": 2.0})

    def test_asr_logger_filters_high_frequency_events_and_writes_jsonl(self) -> None:
        event_filter = _EventFilter(write_segment_events=False, audio_seen_min_interval_s=10.0)

        self.assertTrue(event_filter.should_skip({"type": "segment"}))
        self.assertFalse(event_filter.should_skip({"type": "audio_seen", "stream": "mic", "ts": 100.0}))
        self.assertTrue(event_filter.should_skip({"type": "audio_seen", "stream": "mic", "ts": 105.0}))

        with tempfile.TemporaryDirectory() as tmp:
            logger = ASRLogger(
                root=Path(tmp),
                session_id="unit",
                language="en",
                max_bytes=100_000,
                backup_count=1,
                write_segment_events=True,
            )
            logger.write({"type": "segment", "text": "kept"})
            logger.write("ignored")  # type: ignore[arg-type]
            logger.close()

            self.assertTrue(logger.path.exists())
            self.assertIn('"segment"', logger.path.read_text(encoding="utf-8"))

    def test_faster_whisper_adapter_transcribes_words_and_closes_model(self) -> None:
        class _Word:
            word = " hello "
            start = 0.1
            end = 0.2

        class _Segment:
            words = [_Word()]

        class _Model:
            def __init__(self) -> None:
                self.model = SimpleNamespace(unload_model=lambda: setattr(self, "unloaded", True))
                self.calls = []
                self.unloaded = False

            def transcribe(self, samples, **kwargs):  # noqa: ANN001, ANN003
                self.calls.append((samples, kwargs))
                return [_Segment()], object()

        adapter = FasterWhisperASR.__new__(FasterWhisperASR)
        adapter.language = "en"
        adapter._model = _Model()

        words = adapter.transcribe_words(np.ones((2, 2), dtype=np.float32))
        adapter.close()

        self.assertEqual(words, [{"text": "hello", "start": 0.1, "end": 0.2}])
        self.assertTrue(adapter._model.calls[0][1]["word_timestamps"])
        self.assertTrue(adapter._model.unloaded)


if __name__ == "__main__":
    unittest.main()
