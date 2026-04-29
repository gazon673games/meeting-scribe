from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from asr.application.pipeline_config import ASRPipelineSettings, build_diarization_config
from diarization.infrastructure.diar_backend_sherpa import SherpaOnnxSpeakerEmbeddingBackend
from diarization.infrastructure.diarizer import OnlineDiarizer
from interface.session_controller import _asr_settings_from_params


class _FakeSherpaConfig:
    def __init__(self, *, model: str, num_threads: int = 1, provider: str = "cpu") -> None:
        self.model = model
        self.num_threads = num_threads
        self.provider = provider

    def validate(self) -> bool:
        return True


class _FakeSherpaStream:
    def accept_waveform(self, *, sample_rate: int, waveform) -> None:  # noqa: ANN001
        self.sample_rate = sample_rate
        self.waveform = waveform

    def input_finished(self) -> None:
        self.finished = True


class _FakeSherpaExtractor:
    def __init__(self, config: _FakeSherpaConfig) -> None:
        self.config = config

    def create_stream(self) -> _FakeSherpaStream:
        return _FakeSherpaStream()

    def is_ready(self, stream: _FakeSherpaStream) -> bool:
        return bool(getattr(stream, "finished", False))

    def compute(self, stream: _FakeSherpaStream) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def _install_fake_sherpa_module() -> object:
    previous = sys.modules.get("sherpa_onnx")
    sys.modules["sherpa_onnx"] = types.SimpleNamespace(
        SpeakerEmbeddingExtractorConfig=_FakeSherpaConfig,
        SpeakerEmbeddingExtractor=_FakeSherpaExtractor,
    )
    return previous


def _restore_sherpa_module(previous: object) -> None:
    if previous is None:
        sys.modules.pop("sherpa_onnx", None)
    else:
        sys.modules["sherpa_onnx"] = previous


class SherpaOnnxDiarizerTests(unittest.TestCase):
    def test_backend_computes_embedding_with_local_model_path(self) -> None:
        previous = _install_fake_sherpa_module()
        try:
            with tempfile.TemporaryDirectory() as raw_dir:
                model_path = Path(raw_dir) / "speaker.onnx"
                model_path.write_bytes(b"fake")

                backend = SherpaOnnxSpeakerEmbeddingBackend(
                    model_path=str(model_path),
                    provider="cpu",
                    num_threads=2,
                )

                embedding = backend.embed(np.ones(32000, dtype=np.float32))
        finally:
            _restore_sherpa_module(previous)

        self.assertEqual(embedding.tolist(), [1.0, 0.0, 0.0])

    def test_online_diarizer_uses_sherpa_embedding_backend(self) -> None:
        previous = _install_fake_sherpa_module()
        try:
            with tempfile.TemporaryDirectory() as raw_dir:
                model_path = Path(raw_dir) / "speaker.onnx"
                model_path.write_bytes(b"fake")

                diarizer = OnlineDiarizer(
                    backend="sherpa_onnx",
                    sherpa_model_path=str(model_path),
                    sherpa_provider="cpu",
                    sherpa_num_threads=1,
                )

                label, n_speakers, best_sim, created = diarizer.assign_with_debug(
                    np.ones(32000, dtype=np.float32),
                    ts=10.0,
                )
        finally:
            _restore_sherpa_module(previous)

        self.assertEqual(label, "S1")
        self.assertEqual(n_speakers, 1)
        self.assertEqual(best_sim, 1.0)
        self.assertTrue(created)

    def test_missing_sherpa_model_path_is_reported_as_diarizer_error(self) -> None:
        diarizer = OnlineDiarizer(backend="sherpa_onnx", sherpa_model_path="")

        label, _, _, created = diarizer.assign_with_debug(np.ones(32000, dtype=np.float32), ts=10.0)

        self.assertEqual(label, "S_ERR:RuntimeError")
        self.assertFalse(created)
        self.assertIn("model path is required", diarizer.last_error() or "")

    def test_session_params_accept_sherpa_backend_settings(self) -> None:
        settings = _asr_settings_from_params(
            {
                "diarizationEnabled": True,
                "diarBackend": "sherpa_onnx",
                "diarSherpaEmbeddingModelPath": "models/speaker.onnx",
                "diarSherpaProvider": "cpu",
                "diarSherpaNumThreads": 3,
            }
        )

        self.assertTrue(settings.diarization_enabled)
        self.assertEqual(settings.diar_backend, "sherpa_onnx")
        self.assertEqual(settings.diar_sherpa_embedding_model_path, "models/speaker.onnx")
        self.assertEqual(settings.diar_sherpa_provider, "cpu")
        self.assertEqual(settings.diar_sherpa_num_threads, 3)

    def test_pipeline_settings_build_sherpa_diarization_config(self) -> None:
        config = build_diarization_config(
            ASRPipelineSettings(
                diar_backend="sherpa_onnx",
                diar_sherpa_embedding_model_path="models/speaker.onnx",
                diar_sherpa_provider="cpu",
                diar_sherpa_num_threads=2,
            )
        )

        self.assertEqual(config.backend, "sherpa_onnx")
        self.assertEqual(config.sherpa_embedding_model_path, "models/speaker.onnx")
        self.assertEqual(config.sherpa_provider, "cpu")
        self.assertEqual(config.sherpa_num_threads, 2)


if __name__ == "__main__":
    unittest.main()
