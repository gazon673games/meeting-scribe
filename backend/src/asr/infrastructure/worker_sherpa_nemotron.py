from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

from application.native_asr_models import NEMOTRON_REQUIRED_FILES
from asr.domain.recognition import RecognitionResult, RecognizedWord
from asr.infrastructure.sherpa_runtime import (
    report_cpu_execution,
    sherpa_cpu_threads,
)

_logger = logging.getLogger(__name__)


_LANGUAGE_ALIASES = {
    'russian': 'ru',
    'ru-ru': 'ru',
    'english': 'en',
    'en-us': 'en',
    'en-gb': 'en',
}


class SherpaNemotronASR:
    """Stateful sherpa-onnx adapter for Nemotron 3.5 streaming ASR."""

    def __init__(
        self,
        *,
        model_dir: str | Path,
        model_name: str,
        language: str | None,
        device: str,
        compute_type: str,
        cpu_threads: int,
        num_workers: int,
        beam_size: int,
        initial_prompt: str | None,
        hotwords: str | None,
    ) -> None:
        del (
            compute_type,
            num_workers,
            beam_size,
            initial_prompt,
            hotwords,
        )
        try:
            import sherpa_onnx  # type: ignore
        except (ImportError, OSError) as error:
            raise RuntimeError(
                'sherpa-onnx 1.13.4+ is required for Nemotron 3.5 streaming ASR'
            ) from error

        self._sherpa = sherpa_onnx
        self._root = Path(model_dir).expanduser().resolve()
        missing = [
            name
            for name in NEMOTRON_REQUIRED_FILES
            if not (self._root / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f'Nemotron model is incomplete in {self._root}: {", ".join(missing)}'
            )

        self.device = 'cpu'
        self.cpu_threads = sherpa_cpu_threads(cpu_threads)
        self.num_workers = 1
        report_cpu_execution(model_name, device)
        self._language = _normalize_language(language)
        self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(self._root / 'tokens.txt'),
            encoder=str(self._root / 'encoder.int8.onnx'),
            decoder=str(self._root / 'decoder.int8.onnx'),
            joiner=str(self._root / 'joiner.int8.onnx'),
            num_threads=self.cpu_threads,
            provider='cpu',
            model_type='nemo_transducer',
        )
        self._streams: dict[str, Any] = {}

    def close(self) -> None:
        self._streams.clear()
        self._recognizer = None

    def transcribe_stream_chunk(
        self,
        stream_id: str,
        audio_16k_mono: np.ndarray,
        *,
        is_final: bool,
    ) -> RecognitionResult:
        recognizer = self._require_recognizer()
        stream = self._streams.get(stream_id)
        if stream is None:
            stream = recognizer.create_stream()
            stream.set_option('language', self._language)
            self._streams[stream_id] = stream

        try:
            samples = np.asarray(audio_16k_mono, dtype=np.float32).reshape(-1)
            if samples.size:
                stream.accept_waveform(16000, samples)
            if is_final:
                stream.input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            return _result_dict(recognizer, stream)
        finally:
            if is_final:
                self._streams.pop(stream_id, None)

    def transcribe(
        self, audio_16k_mono: np.ndarray, *, beam_size: int | None = None
    ) -> RecognitionResult:
        del beam_size
        return self.transcribe_stream_chunk(
            '__batch__',
            audio_16k_mono,
            is_final=True,
        )

    def transcribe_words(
        self, audio_16k_mono: np.ndarray
    ) -> list[RecognizedWord]:
        result = self.transcribe(audio_16k_mono)
        return _text_to_words(
            str(result.get('text') or ''), result.get('timestamps')
        )

    def _require_recognizer(self) -> Any:
        if self._recognizer is None:
            raise RuntimeError('Nemotron recognizer is closed')
        return self._recognizer


def _normalize_language(language: str | None) -> str:
    value = str(language or 'auto').strip().lower()
    return _LANGUAGE_ALIASES.get(value, value or 'auto')


def _result_dict(recognizer: Any, stream: Any) -> RecognitionResult:
    try:
        raw = json.loads(recognizer.get_result_as_json_string(stream))
        if isinstance(raw, dict):
            raw['text'] = str(raw.get('text') or '')
            return cast(RecognitionResult, raw)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        _logger.debug(
            'Best-effort operation failed in _result_dict', exc_info=True
        )
    return {'text': str(recognizer.get_result(stream) or '')}


def _text_to_words(text: str, timestamps: Any = None) -> list[RecognizedWord]:
    parts = [part for part in str(text or '').strip().split() if part]
    times = [
        float(value)
        for value in (timestamps or [])
        if isinstance(value, (int, float))
    ]
    if not parts:
        return []
    end_s = max(times, default=0.0)
    step = end_s / max(1, len(parts))
    return [
        {'text': word, 'start': index * step, 'end': (index + 1) * step}
        for index, word in enumerate(parts)
    ]
