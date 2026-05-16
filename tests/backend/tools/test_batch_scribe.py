from __future__ import annotations

import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import batch_scribe
from tools.batch_scribe_parts import entrypoint


class BatchScribeEntryPointTests(unittest.TestCase):
    def test_scribe_to_srt_passes_one_time_options_and_writes_srt(self) -> None:
        created: list["FakeScribe"] = []

        class FakeScribe:
            def __init__(self, profile, **kwargs) -> None:  # noqa: ANN001
                self.profile = profile
                self.kwargs = kwargs
                created.append(self)

            def __enter__(self) -> "FakeScribe":
                return self

            def __exit__(self, *_) -> None:  # noqa: ANN002
                return None

            def process(self, input_path: Path, tmp_dir: Path, *, word_by_word: bool = False) -> list[dict]:
                self.input_path = Path(input_path)
                self.tmp_dir_exists = Path(tmp_dir).exists()
                self.word_by_word = word_by_word
                return [
                    {"t0": 0.0, "t1": 0.4, "text": "hello", "unit": "word"},
                    {"t0": 0.4, "t1": 0.8, "text": "world", "unit": "word"},
                ]

        with tempfile.TemporaryDirectory() as raw_root, patch.object(entrypoint, "Scribe", FakeScribe):
            root = Path(raw_root)
            source = root / "meeting.mp4"
            output = root / "meeting.srt"
            source.write_bytes(b"fake video")

            result = batch_scribe.scribe_to_srt(
                source,
                output,
                profile_name="Custom",
                model="small",
                device="cpu",
                language="en",
                compute_type="int8",
                beam_size=2,
                cpu_threads=3,
                num_workers=4,
                temperature=0.2,
                asr_options={"patience": 1.5},
                word_by_word=True,
                diar=True,
                diar_backend="nemo",
            )

            written = output.read_text(encoding="utf-8")

        self.assertEqual(result.output_path, output)
        self.assertEqual(result.output_format, "srt")
        self.assertIn("00:00:00,000 --> 00:00:00,400", written)
        self.assertIn("hello", written)
        self.assertEqual(created[0].input_path, source)
        self.assertTrue(created[0].tmp_dir_exists)
        self.assertTrue(created[0].word_by_word)
        self.assertEqual(created[0].profile.model_name, "small")
        self.assertEqual(created[0].profile.device, "cpu")
        self.assertEqual(created[0].profile.compute_type, "int8")
        self.assertEqual(created[0].profile.beam_size, 2)
        self.assertEqual(created[0].profile.language, "en")
        self.assertEqual(created[0].profile.cpu_threads, 3)
        self.assertEqual(created[0].profile.num_workers, 4)
        self.assertEqual(created[0].profile.temperature, 0.2)
        self.assertEqual(created[0].profile.extra_transcribe_options, {"patience": 1.5})
        self.assertTrue(created[0].kwargs["diar"])
        self.assertEqual(created[0].kwargs["diar_backend"], "nemo")

    def test_word_by_word_transcription_uses_word_timestamps(self) -> None:
        class FakeModel:
            def transcribe(self, wav_path: str, **kwargs):  # noqa: ANN001
                self.wav_path = wav_path
                self.kwargs = kwargs
                words = [
                    types.SimpleNamespace(start=0.1, end=0.3, word=" hello "),
                    types.SimpleNamespace(start=0.3, end=0.6, word="world"),
                ]
                return [types.SimpleNamespace(start=0.0, end=0.7, text="hello world", words=words)], object()

        profile = types.SimpleNamespace(
            device="cpu",
            model_name="small",
            compute_type="int8",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=True,
            language="en",
            initial_prompt=None,
            temperature=0.1,
            extra_transcribe_options={"patience": 2},
        )
        scribe = batch_scribe.Scribe(profile)
        scribe._model = FakeModel()

        records = scribe._transcribe(Path("audio.wav"), word_by_word=True)

        self.assertEqual(
            records,
            [
                {"t0": 0.1, "t1": 0.3, "text": "hello", "unit": "word"},
                {"t0": 0.3, "t1": 0.6, "text": "world", "unit": "word"},
            ],
        )
        self.assertEqual(scribe._model.kwargs["patience"], 2)
        self.assertEqual(scribe._model.kwargs["temperature"], 0.1)
        self.assertTrue(scribe._model.kwargs["word_timestamps"])
        self.assertFalse(scribe._model.kwargs["without_timestamps"])

    def test_parse_asr_options_coerces_cli_values(self) -> None:
        options = batch_scribe._parse_asr_options(["patience=1.5", "best_of=3", "clip_timestamps=false"])

        self.assertEqual(options, {"patience": 1.5, "best_of": 3, "clip_timestamps": False})

    def test_model_load_failure_reports_missing_model_hint(self) -> None:
        class FailingWhisperModel:
            def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
                raise OSError("not found")

        faster_whisper = types.ModuleType("faster_whisper")
        faster_whisper.WhisperModel = FailingWhisperModel
        profile = types.SimpleNamespace(
            device="cpu",
            model_name="missing-model",
            compute_type="int8",
            cpu_threads=None,
            num_workers=None,
        )

        with patch.dict(sys.modules, {"faster_whisper": faster_whisper}):
            with self.assertRaisesRegex(RuntimeError, "Unable to load ASR model 'missing-model'"):
                with batch_scribe.Scribe(profile):
                    pass


if __name__ == "__main__":
    unittest.main()
