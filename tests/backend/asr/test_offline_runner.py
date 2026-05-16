from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from asr.infrastructure.offline_runner import OfflineProfile, OfflineRunner


class OfflineRunnerTests(unittest.TestCase):
    def test_run_transcribes_segments_writes_outputs_and_unloads_model(self) -> None:
        created_models: list["FakeWhisperModel"] = []

        class Segment:
            def __init__(self, start: float, end: float, text: str) -> None:
                self.start = start
                self.end = end
                self.text = text

        class FakeModelHandle:
            def __init__(self) -> None:
                self.unloaded = False

            def unload_model(self) -> None:
                self.unloaded = True

        class FakeWhisperModel:
            def __init__(self, model_name: str, *, device: str, compute_type: str) -> None:
                self.model_name = model_name
                self.device = device
                self.compute_type = compute_type
                self.model = FakeModelHandle()
                created_models.append(self)

            def transcribe(self, wav_path: str, **kwargs):  # noqa: ANN001
                self.wav_path = wav_path
                self.kwargs = kwargs
                return [
                    Segment(0.0, 1.0, " hello "),
                    Segment(1.0, 2.0, ""),
                    Segment(2.0, 3.5, "world"),
                ], None

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = FakeWhisperModel

        with tempfile.TemporaryDirectory() as raw_root, patch.dict(sys.modules, {"faster_whisper": faster_whisper}):
            root = Path(raw_root)
            wav_path = root / "input.wav"
            wav_path.write_bytes(b"fake")
            out_txt = root / "out" / "transcript.txt"
            out_jsonl = root / "out" / "segments.jsonl"

            result = OfflineRunner(project_root=root).run(
                wav_path,
                out_txt=out_txt,
                out_jsonl=out_jsonl,
                profile=OfflineProfile(model_name="large-v3", device="cpu", compute_type="int8", beam_size=4, language="ru"),
            )

            out_text = out_txt.read_text(encoding="utf-8")
            records = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result, out_txt)
        self.assertEqual(out_text, "hello\nworld\n")
        self.assertEqual([record["text"] for record in records], ["hello", "world"])
        self.assertEqual(created_models[0].model_name, "large-v3")
        self.assertEqual(created_models[0].kwargs["language"], "ru")
        self.assertEqual(created_models[0].kwargs["beam_size"], 4)
        self.assertTrue(created_models[0].model.unloaded)

    def test_run_reports_missing_faster_whisper_with_install_hint(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root, patch.dict(sys.modules, {"faster_whisper": None}):
            root = Path(raw_root)
            with self.assertRaisesRegex(RuntimeError, "Offline pass requires faster-whisper"):
                OfflineRunner(project_root=root).run(root / "missing.wav", out_txt=root / "out.txt")


if __name__ == "__main__":
    unittest.main()
