from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from application.native_asr_models import NEMOTRON_REQUIRED_FILES


class SherpaNemotronASR:
    """Stateful sherpa-onnx adapter for Nemotron 3.5 streaming ASR."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        model_name: str,
        language: Optional[str],
        device: str,
        compute_type: str,
        cpu_threads: int,
        num_workers: int,
        beam_size: int,
        initial_prompt: Optional[str],
        hotwords: Optional[str],
    ) -> None:
        del model_name, device, compute_type, num_workers, beam_size, initial_prompt, hotwords
        try:
            import sherpa_onnx  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "sherpa-onnx 1.13.4+ is required for Nemotron 3.5 streaming ASR"
            ) from exc

        self._sherpa = sherpa_onnx
        self._root = Path(model_dir).expanduser().resolve()
        missing = [name for name in NEMOTRON_REQUIRED_FILES if not (self._root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Nemotron model is incomplete in {self._root}: {', '.join(missing)}")

        threads = int(cpu_threads) if int(cpu_threads) > 0 else min(4, max(1, int(os.cpu_count() or 1)))
        self._language = _normalize_language(language)
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(self._root / "tokens.txt"),
            encoder=str(self._root / "encoder.int8.onnx"),
            decoder=str(self._root / "decoder.int8.onnx"),
            joiner=str(self._root / "joiner.int8.onnx"),
            num_threads=threads,
            provider="cpu",
            model_type="nemo_transducer",
        )
        self._streams: Dict[str, Any] = {}

    def close(self) -> None:
        self._streams.clear()
        self._recognizer = None

    def transcribe_stream_chunk(
        self,
        stream_id: str,
        audio_16k_mono: np.ndarray,
        *,
        is_final: bool,
    ) -> Dict[str, Any]:
        recognizer = self._require_recognizer()
        stream = self._streams.get(stream_id)
        if stream is None:
            stream = recognizer.create_stream()
            stream.set_option("language", self._language)
            self._streams[stream_id] = stream

        samples = np.asarray(audio_16k_mono, dtype=np.float32).reshape(-1)
        if samples.size:
            stream.accept_waveform(16000, samples)
        if is_final:
            stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        result = _result_dict(recognizer, stream)
        if is_final:
            self._streams.pop(stream_id, None)
        return result

    def transcribe(self, audio_16k_mono: np.ndarray, *, beam_size: Optional[int] = None) -> Dict[str, Any]:
        del beam_size
        return self.transcribe_stream_chunk(
            "__batch__",
            audio_16k_mono,
            is_final=True,
        )

    def transcribe_words(self, audio_16k_mono: np.ndarray) -> List[Dict[str, Any]]:
        result = self.transcribe(audio_16k_mono)
        return _text_to_words(str(result.get("text") or ""), result.get("timestamps"))

    def _require_recognizer(self) -> Any:
        if self._recognizer is None:
            raise RuntimeError("Nemotron recognizer is closed")
        return self._recognizer


def _normalize_language(language: Optional[str]) -> str:
    value = str(language or "auto").strip().lower()
    if value in {"russian", "ru-ru"}:
        return "ru"
    if value in {"english", "en-us", "en-gb"}:
        return "en"
    return value or "auto"


def _result_dict(recognizer: Any, stream: Any) -> Dict[str, Any]:
    try:
        raw = json.loads(recognizer.get_result_as_json_string(stream))
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {"text": str(recognizer.get_result(stream) or "")}


def _text_to_words(text: str, timestamps: Any = None) -> List[Dict[str, Any]]:
    parts = [part for part in str(text or "").strip().split() if part]
    times = [float(value) for value in (timestamps or []) if isinstance(value, (int, float))]
    if not parts:
        return []
    end_s = max(times, default=0.0)
    step = end_s / max(1, len(parts))
    return [
        {"text": word, "start": index * step, "end": (index + 1) * step}
        for index, word in enumerate(parts)
    ]
