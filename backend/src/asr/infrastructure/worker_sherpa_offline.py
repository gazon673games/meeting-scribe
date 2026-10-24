from __future__ import annotations

from pathlib import Path

import numpy as np

from application.native_asr_models import (
    PARAKEET_MODEL_ID,
    SHERPA_OFFLINE_MODEL_IDS,
    native_asr_required_files,
)
from asr.domain.recognition import RecognitionResult, RecognizedWord
from asr.infrastructure.sherpa_runtime import (
    report_cpu_execution,
    sherpa_cpu_threads,
)


class SherpaOfflineASR:
    """Segment-based Parakeet TDT and Qwen3-ASR using local ONNX weights."""

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
        del language, compute_type, num_workers, beam_size, initial_prompt
        if model_name not in SHERPA_OFFLINE_MODEL_IDS:
            raise ValueError(f'Unsupported offline ONNX model: {model_name}')
        import sherpa_onnx

        root = Path(model_dir).expanduser().resolve()
        missing = [
            name
            for name in native_asr_required_files(model_name)
            if not (root / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f'ASR model is incomplete in {root}: {", ".join(missing)}'
            )
        self.device = 'cpu'
        self.cpu_threads = sherpa_cpu_threads(cpu_threads)
        self.num_workers = 1
        report_cpu_execution(model_name, device)
        if model_name == PARAKEET_MODEL_ID:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(root / 'encoder.int8.onnx'),
                decoder=str(root / 'decoder.int8.onnx'),
                joiner=str(root / 'joiner.int8.onnx'),
                tokens=str(root / 'tokens.txt'),
                model_type='nemo_transducer',
                num_threads=self.cpu_threads,
                provider=self.device,
            )
        else:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
                conv_frontend=str(root / 'conv_frontend.onnx'),
                encoder=str(root / 'encoder.int8.onnx'),
                decoder=str(root / 'decoder.int8.onnx'),
                tokenizer=str(root / 'tokenizer'),
                hotwords=hotwords or '',
                max_total_len=2048,
                max_new_tokens=512,
                num_threads=self.cpu_threads,
                provider=self.device,
            )

    def close(self) -> None:
        self._recognizer = None

    def transcribe(
        self, audio_16k_mono: np.ndarray, *, beam_size: int | None = None
    ) -> RecognitionResult:
        del beam_size
        if self._recognizer is None:
            raise RuntimeError('Offline ONNX recognizer is closed')
        stream = self._recognizer.create_stream()
        samples = np.asarray(audio_16k_mono, dtype=np.float32).reshape(-1)
        stream.accept_waveform(16000, samples)
        self._recognizer.decode_stream(stream)
        return {
            'text': str(stream.result.text).strip(),
            'timestamps': [float(value) for value in stream.result.timestamps],
        }

    def transcribe_words(
        self, audio_16k_mono: np.ndarray
    ) -> list[RecognizedWord]:
        # Prefix confirmation needs word-shaped records even when the SDK
        # exposes subword timestamps. These intervals are approximate.
        result = self.transcribe(audio_16k_mono)
        words = result['text'].split()
        step_s = np.asarray(audio_16k_mono).size / 16000 / max(1, len(words))
        return [
            {'text': word, 'start': index * step_s, 'end': (index + 1) * step_s}
            for index, word in enumerate(words)
        ]
