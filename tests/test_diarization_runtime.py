from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from asr.domain.segments import Segment
from diarization.application.diarization import DiarizationConfig
from diarization.domain.segments import DiarSegment
from diarization.infrastructure.diarization_runtime import (
    DefaultDiarizationRuntimeFactory,
    DiarizationRuntime,
    _embedding_backend_name,
)


class _Audio:
    def __init__(self, samples) -> None:  # noqa: ANN001
        self.samples = np.asarray(samples, dtype=np.float32)


class _OnlineDiarizer:
    def __init__(self, speaker: str = "S1") -> None:
        self.speaker = speaker
        self.assigned: list[np.ndarray] = []

    def assign_with_debug(self, audio, ts):  # noqa: ANN001, ANN201
        self.assigned.append(np.asarray(audio))
        return self.speaker, 2, 0.91, self.speaker == "S2"

    def last_error(self) -> str:
        return "embed failed"

    def clusters_snapshot(self):  # noqa: ANN201
        return [
            {"label": "S1", "centroid": np.array([1.0, 0.0]), "count": 2, "last_ts": 3.0},
            {"label": "", "centroid": np.array([0.0, 1.0])},
            {"label": "bad", "centroid": None},
        ]


class _Pyannote:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def diarize(self, ring, t_offset):  # noqa: ANN001, ANN201
        if self.fail:
            raise RuntimeError("diar failed")
        return [DiarSegment(t_offset, t_offset + 10.0, "P1")]


def _config(**kwargs) -> DiarizationConfig:  # noqa: ANN003
    values = {
        "enabled": True,
        "backend": "online",
        "sim_threshold": 0.7,
        "min_segment_s": 0.5,
        "window_s": 2.0,
        "chunk_s": 6.0,
        "step_s": 1.0,
        "device": "cpu",
        "source_speaker_labels": {"mic": "Speaker Mic"},
        "sherpa_num_threads": 2,
    }
    values.update(kwargs)
    return DiarizationConfig(**values)


def _segment(stream: str = "mic") -> Segment:
    return Segment(stream=stream, t_start=1.0, t_end=2.5, audio=_Audio([0.1, 0.2]), enqueue_ts=0.0)


class DiarizationRuntimeTests(unittest.TestCase):
    def test_disabled_runtime_uses_source_fallback_and_empty_snapshot(self) -> None:
        runtime = DiarizationRuntime(
            config=_config(enabled=False),
            online_diarizer_factory=lambda **kwargs: _OnlineDiarizer(),
            pyannote_diarizer_factory=lambda **kwargs: _Pyannote(),
        )

        runtime.ensure_stream("mic")
        runtime.update_ring("mic", 1.0, np.ones(4))

        self.assertEqual(runtime.speaker_for_segment(_segment(), lambda event: None), "Speaker Mic")
        self.assertEqual(runtime.identity_snapshot(), {})

    def test_online_runtime_assigns_speaker_logs_debug_and_exports_identity(self) -> None:
        created: list[dict] = []

        def online_factory(**kwargs):  # noqa: ANN003
            created.append(kwargs)
            return _OnlineDiarizer("S2")

        events: list[dict] = []
        runtime = DiarizationRuntime(
            config=_config(backend="sherpa_onnx", sherpa_embedding_model_path="speaker.onnx", sherpa_provider="cpu"),
            online_diarizer_factory=online_factory,
            pyannote_diarizer_factory=lambda **kwargs: _Pyannote(),
        )

        runtime.init_backend(events.append)
        speaker = runtime.speaker_for_segment(_segment(), events.append)

        self.assertEqual(speaker, "S2")
        self.assertEqual(created[0]["backend"], "sherpa_onnx")
        self.assertEqual(events[0]["type"], "diar_init_ok")
        self.assertEqual(events[-1]["type"], "diar_debug")
        snapshot = runtime.identity_snapshot()
        self.assertEqual(snapshot["mic"]["S1"]["embedding_dim"], 2)
        self.assertEqual(snapshot["mic"]["S1"]["embedding_model"], "sherpa_onnx")

    def test_online_runtime_reports_embed_errors_and_assignment_exceptions(self) -> None:
        events: list[dict] = []
        err_runtime = DiarizationRuntime(
            config=_config(),
            online_diarizer_factory=lambda **kwargs: _OnlineDiarizer("S_ERR_BACKEND"),
            pyannote_diarizer_factory=lambda **kwargs: _Pyannote(),
        )
        self.assertEqual(err_runtime.speaker_for_segment(_segment(), events.append), "Speaker Mic")
        self.assertEqual(events[0]["where"], "diar_embed")

        failing_runtime = DiarizationRuntime(
            config=_config(),
            online_diarizer_factory=lambda **kwargs: SimpleNamespace(assign_with_debug=lambda *args, **kw: (_ for _ in ()).throw(RuntimeError("boom"))),
            pyannote_diarizer_factory=lambda **kwargs: _Pyannote(),
        )
        events.clear()
        self.assertEqual(failing_runtime.speaker_for_segment(_segment(), events.append), "Speaker Mic")
        self.assertEqual(events[0]["where"], "diar_assign")

    def test_pyannote_runtime_updates_ring_timeline_and_falls_back_on_init_error(self) -> None:
        events: list[dict] = []
        runtime = DiarizationRuntime(
            config=_config(backend="pyannote", chunk_s=6.0, step_s=0.5),
            online_diarizer_factory=lambda **kwargs: _OnlineDiarizer(),
            pyannote_diarizer_factory=lambda **kwargs: _Pyannote(),
        )

        runtime.init_backend(events.append)
        self.assertEqual(events[0]["backend"], "pyannote")
        runtime.update_ring("mic", 7.0, np.ones(16000 * 7, dtype=np.float32))
        self.assertLessEqual(runtime._ring_samples["mic"], 16000 * 6)

        with patch("diarization.infrastructure.diarization_runtime.time.time", return_value=20.0):
            self.assertEqual(runtime.speaker_for_segment(_segment(), events.append), "P1")
        self.assertTrue(runtime._timeline["mic"])
        self.assertEqual(runtime._ring_array("mic").dtype, np.float32)

        failing_init = DiarizationRuntime(
            config=_config(backend="pyannote"),
            online_diarizer_factory=lambda **kwargs: _OnlineDiarizer(),
            pyannote_diarizer_factory=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("missing token")),
        )
        fallback_events: list[dict] = []
        failing_init.init_backend(fallback_events.append)
        self.assertEqual(failing_init.backend, "nemo")
        self.assertIn("diar_fallback", [event["type"] for event in fallback_events])

    def test_pyannote_failure_disables_runtime_and_factory_creates_runtime(self) -> None:
        events: list[dict] = []
        runtime = DiarizationRuntime(
            config=_config(backend="pyannote"),
            online_diarizer_factory=lambda **kwargs: _OnlineDiarizer(),
            pyannote_diarizer_factory=lambda **kwargs: _Pyannote(fail=True),
        )
        runtime.init_backend(events.append)
        runtime.update_ring("mic", 10.0, np.ones(16000 * 7, dtype=np.float32))

        with patch("diarization.infrastructure.diarization_runtime.time.time", return_value=20.0):
            self.assertEqual(runtime.speaker_for_segment(_segment(), events.append), "Speaker Mic")
        self.assertFalse(runtime.enabled)
        self.assertEqual(events[-1]["where"], "diar_run")

        with tempfile.TemporaryDirectory() as raw_root:
            factory = DefaultDiarizationRuntimeFactory(
                online_diarizer_factory=lambda **kwargs: _OnlineDiarizer(),
                pyannote_diarizer_factory=lambda **kwargs: _Pyannote(),
            )
            made = factory(config=_config(temp_dir=Path(raw_root)))
        self.assertIsInstance(made, DiarizationRuntime)
        self.assertEqual(_embedding_backend_name("nemo"), "nemo")
        self.assertEqual(_embedding_backend_name("online"), "resemblyzer")


if __name__ == "__main__":
    unittest.main()
