from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from diarization.infrastructure.diar_backend_nemo import NeMoTitaNetEmbedder, _to_float32_mono
from diarization.infrastructure.diar_backend_pyannote import PyannoteDiarizer
from diarization.infrastructure import diarizer


class _FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    @staticmethod
    def device(name: str) -> str:
        return f"device:{name}"

    @staticmethod
    def from_numpy(array):  # noqa: ANN001
        class Tensor:
            def __init__(self, value) -> None:  # noqa: ANN001
                self.value = value

            def unsqueeze(self, dim: int):  # noqa: ANN001
                self.dim = dim
                return self

        return Tensor(array)


class DiarizationBackendTests(unittest.TestCase):
    def test_online_diarizer_backends_and_cluster_helpers_assign_speakers(self) -> None:
        self.assertAlmostEqual(diarizer._cos_sim(np.array([1.0, 0.0]), np.array([1.0, 0.0])), 1.0)

        class FakeEncoder:
            def embed_utterance(self, audio):  # noqa: ANN001
                return np.asarray([1.0, 0.0], dtype=np.float32)

        resemblyzer_module = types.ModuleType("resemblyzer")
        resemblyzer_module.VoiceEncoder = lambda: FakeEncoder()
        with patch.dict(sys.modules, {"resemblyzer": resemblyzer_module}):
            backend = diarizer._ResemblyzerBackend()
            self.assertEqual(backend.embed(np.ones((2, 2), dtype=np.float32)).tolist(), [1.0, 0.0])

        class FakeNeMoEmbedder:
            def __init__(self, *, device, temp_dir):  # noqa: ANN001
                self.device = device
                self.temp_dir = temp_dir

            def embed_16k(self, audio, *, sample_rate):  # noqa: ANN001
                return np.asarray([0.0, 1.0], dtype=np.float32)

        with patch("diarization.infrastructure.diar_backend_nemo.NeMoTitaNetEmbedder", FakeNeMoEmbedder):
            nemo = diarizer._NeMoBackend(device="cpu", temp_dir=Path("tmp"))
            self.assertEqual(nemo.embed(np.ones(16000, dtype=np.float32)).tolist(), [0.0, 1.0])

        online = diarizer.OnlineDiarizer(min_segment_s=0.0, backend="resemblyzer", similarity_threshold=0.5)
        online._backend = SimpleNamespace(embed=lambda audio: np.asarray([1.0, 0.0], dtype=np.float32))
        self.assertEqual(online.assign(np.ones(16000, dtype=np.float32), ts=1.0)[0], "S1")
        self.assertEqual(online.assign(np.ones(16000, dtype=np.float32), ts=2.0)[0], "S1")
        self.assertEqual(online.clusters_snapshot()[0]["label"], "S1")

    def test_pyannote_diarizer_loads_pipeline_and_offsets_segments(self) -> None:
        class FakePipeline:
            moved_to = None

            @classmethod
            def from_pretrained(cls, name: str):  # noqa: ANN001
                cls.name = name
                return cls()

            def to(self, device):  # noqa: ANN001
                FakePipeline.moved_to = device

            def __call__(self, payload):  # noqa: ANN001
                self.payload = payload

                class Diarization:
                    def itertracks(self, *, yield_label: bool):  # noqa: ARG002
                        yield types.SimpleNamespace(start=1.5, end=2.25), None, "S1"
                        yield types.SimpleNamespace(start=3.0, end=3.5), None, "S2"

                return Diarization()

        audio_module = types.ModuleType("pyannote.audio")
        audio_module.Pipeline = FakePipeline

        with patch.dict(sys.modules, {"pyannote.audio": audio_module, "torch": _FakeTorch}):
            diarizer = PyannoteDiarizer(device="cuda")
            segments = diarizer.diarize(np.asarray([0.1, 0.2], dtype=np.float32), t_offset=10.0)

        self.assertEqual(FakePipeline.name, "pyannote/speaker-diarization")
        self.assertEqual(FakePipeline.moved_to, "device:cpu")
        self.assertEqual([(segment.t0, segment.t1, segment.speaker) for segment in segments], [(11.5, 12.25, "S1"), (13.0, 13.5, "S2")])
        self.assertEqual(diarizer.diarize(np.asarray([], dtype=np.float32)), [])

    def test_nemo_embedder_writes_temp_wav_and_returns_float32_embedding(self) -> None:
        written_paths: list[str] = []

        class FakeSoundFile(types.ModuleType):
            def write(self, path, data, *, samplerate, subtype):  # noqa: ANN001
                written_paths.append(path)
                self.last_write = {
                    "data": np.asarray(data),
                    "samplerate": samplerate,
                    "subtype": subtype,
                }
                Path(path).write_bytes(b"wav")

        class FakeEmbedding:
            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return np.asarray([1.0, 2.0, 3.0], dtype=np.float64)

        class FakeModel:
            requested_paths: list[str] = []

            @classmethod
            def from_pretrained(cls, *, model_name: str):  # noqa: ANN001
                cls.model_name = model_name
                return cls()

            def to(self, device: str):
                self.device = device
                return self

            def eval(self) -> None:
                self.evaluated = True

            def get_embedding(self, *, audio_file_paths):  # noqa: ANN001
                FakeModel.requested_paths = list(audio_file_paths)
                return [FakeEmbedding()]

        soundfile = FakeSoundFile("soundfile")
        models_module = types.ModuleType("nemo.collections.asr.models")
        models_module.EncDecSpeakerLabelModel = FakeModel
        modules = {
            "soundfile": soundfile,
            "torch": _FakeTorch,
            "nemo": types.ModuleType("nemo"),
            "nemo.collections": types.ModuleType("nemo.collections"),
            "nemo.collections.asr": types.ModuleType("nemo.collections.asr"),
            "nemo.collections.asr.models": models_module,
        }

        with tempfile.TemporaryDirectory() as raw_tmp, patch.dict(sys.modules, modules):
            embedder = NeMoTitaNetEmbedder(device="cuda", model_name="titanet_test", temp_dir=Path(raw_tmp))
            embedding = embedder.embed_16k(np.ones(16000, dtype=np.float32), sample_rate=16000)

            self.assertEqual(embedder.device, "cpu")
            self.assertEqual(FakeModel.model_name, "titanet_test")
            self.assertEqual(embedding.dtype, np.float32)
            self.assertEqual(embedding.tolist(), [1.0, 2.0, 3.0])
            self.assertEqual(FakeModel.requested_paths, written_paths)
            self.assertFalse(Path(written_paths[0]).exists())
            with self.assertRaisesRegex(RuntimeError, "too short"):
                embedder.embed_16k(np.ones(10, dtype=np.float32), sample_rate=16000)

        self.assertEqual(_to_float32_mono([[1, 2], [3, 4]]).tolist(), [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
