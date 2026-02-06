# --- File: D:\work\own\voice2textTest\asr\worker_faster_whisper.py ---
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterable

import numpy as np


@dataclass
class FasterWhisperASR:
    """
    Wrapper around faster-whisper if installed.
    If not installed, raises RuntimeError with actionable message.
    """
    model_name: str = "large-v3"
    language: str = "ru"
    device: str = "cuda"          # "cpu" or "cuda"
    compute_type: str = "int8_float16"
    beam_size: int = 5

    def __post_init__(self) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "faster-whisper is not installed. Install it (and ctranslate2) to enable ASR.\n"
                "Example: pip install faster-whisper\n"
                f"Import error: {type(e).__name__}: {e}"
            )

        self._WhisperModel = WhisperModel
        self._model = self._WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio_16k_mono: np.ndarray) -> Dict[str, Any]:
        """
        audio_16k_mono: float32 mono 16k
        Returns dict with 'text' and optional metadata.
        """
        x = np.asarray(audio_16k_mono, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1).astype(np.float32, copy=False)

        segments, info = self._model.transcribe(
            x,
            language=self.language,
            beam_size=int(self.beam_size),
            vad_filter=False,  # we do VAD ourselves
        )

        text_parts = []
        for s in segments:
            t = getattr(s, "text", "")
            if t:
                text_parts.append(t)

        text = "".join(text_parts).strip()
        out = {
            "text": text,
        }

        # best-effort metadata
        try:
            out["language"] = getattr(info, "language", None)
            out["language_probability"] = getattr(info, "language_probability", None)
        except Exception:
            pass

        return out
