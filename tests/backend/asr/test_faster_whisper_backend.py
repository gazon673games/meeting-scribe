from __future__ import annotations

import sys
import types
import unittest

import numpy as np

from asr.infrastructure.worker_faster_whisper import FasterWhisperASR


class _FakeSegment:
    text = " ok"


class _FakeInfo:
    language = "ru"
    language_probability = 1.0


class _FakeWhisperModel:
    last_init: dict[str, object] = {}
    last_kwargs: dict[str, object] = {}

    def __init__(self, model_name: str, **kwargs: object) -> None:
        self.__class__.last_init = {"model_name": model_name, **kwargs}

    def transcribe(self, audio: np.ndarray, **kwargs: object):  # noqa: ANN201
        self.__class__.last_kwargs = dict(kwargs)
        return [_FakeSegment()], _FakeInfo()


class FasterWhisperBackendTests(unittest.TestCase):
    def test_realtime_decode_uses_low_latency_options(self) -> None:
        previous = sys.modules.get("faster_whisper")
        fake_module = types.SimpleNamespace(WhisperModel=_FakeWhisperModel)
        sys.modules["faster_whisper"] = fake_module  # type: ignore[assignment]
        try:
            asr = FasterWhisperASR(model_name="tiny", language="ru", device="cuda", compute_type="int8_float16")
            result = asr.transcribe(np.zeros((16000,), dtype=np.float32), beam_size=1)
        finally:
            if previous is None:
                sys.modules.pop("faster_whisper", None)
            else:
                sys.modules["faster_whisper"] = previous

        self.assertEqual(result["text"], "ok")
        self.assertEqual(_FakeWhisperModel.last_init["cpu_threads"], 0)
        self.assertEqual(_FakeWhisperModel.last_init["num_workers"], 1)
        self.assertEqual(_FakeWhisperModel.last_kwargs["beam_size"], 1)
        self.assertEqual(_FakeWhisperModel.last_kwargs["best_of"], 1)
        self.assertEqual(_FakeWhisperModel.last_kwargs["temperature"], 0.0)
        self.assertTrue(_FakeWhisperModel.last_kwargs["without_timestamps"])
        self.assertFalse(_FakeWhisperModel.last_kwargs["condition_on_previous_text"])


if __name__ == "__main__":
    unittest.main()
