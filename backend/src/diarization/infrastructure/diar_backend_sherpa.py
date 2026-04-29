from __future__ import annotations

from pathlib import Path

import numpy as np


class SherpaOnnxSpeakerEmbeddingBackend:
    """Speaker embedding backend based on a local sherpa-onnx model file."""

    def __init__(self, *, model_path: str, provider: str = "cpu", num_threads: int = 1) -> None:
        resolved_model_path = _resolve_model_path(model_path)
        try:
            import sherpa_onnx  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Sherpa-ONNX diarization requires the local 'sherpa_onnx' package. "
                "Install it once, then run offline with a downloaded speaker embedding model. "
                f"Import error: {type(exc).__name__}: {exc}"
            ) from exc

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(resolved_model_path),
            num_threads=max(1, int(num_threads)),
            provider=str(provider or "cpu"),
        )
        if hasattr(config, "validate") and not config.validate():
            raise RuntimeError(f"Invalid Sherpa-ONNX speaker embedding config: {config}")

        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        samples = np.asarray(audio_16k, dtype=np.float32)
        if samples.ndim != 1:
            samples = samples.reshape(-1)
        samples = np.ascontiguousarray(samples, dtype=np.float32)

        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=16000, waveform=samples)
        stream.input_finished()

        if hasattr(self._extractor, "is_ready") and not self._extractor.is_ready(stream):
            raise RuntimeError("Sherpa-ONNX speaker embedding extractor is not ready for this segment")

        embedding = self._extractor.compute(stream)
        return np.asarray(embedding, dtype=np.float32)


def _resolve_model_path(model_path: str) -> Path:
    text = str(model_path or "").strip()
    if not text:
        raise RuntimeError("Sherpa-ONNX speaker embedding model path is required")

    path = Path(text).expanduser()
    if not path.is_file():
        raise RuntimeError(f"Sherpa-ONNX speaker embedding model does not exist: {path}")
    return path
