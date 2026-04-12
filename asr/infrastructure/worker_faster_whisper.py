# --- File: asr/worker_faster_whisper.py ---
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np


@dataclass
class FasterWhisperASR:
    """
    Wrapper around faster-whisper.

    Quality knobs:
      - beam_size (higher -> better, slower). Can be overridden per call.
      - temperature (optional, can help on hard audio)
      - initial_prompt (optional, helps with domain terms)
      - condition_on_previous_text (usually True helps coherence)
      - language: "ru"/"en"/... or None for auto-detect
    """
    model_name: str = "large-v3"
    language: Optional[str] = "ru"   # <--- CHANGED: Optional
    device: str = "cuda"            # "cpu" or "cuda"
    compute_type: str = "int8_float16"
    beam_size: int = 5

    temperature: Optional[float] = None
    initial_prompt: Optional[str] = None
    condition_on_previous_text: bool = True

    def __post_init__(self) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "faster-whisper is not installed. Install it (and ctranslate2) to enable ASR.\n"
                "Example: pip install faster-whisper\n"
                f"Import error: {type(e).__name__}: {e}"
            )

        self._model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

        if self.language is not None:
            s = str(self.language).strip().lower()
            self.language = s if s else None

    def transcribe(self, audio_16k_mono: np.ndarray, *, beam_size: Optional[int] = None) -> Dict[str, Any]:
        x = np.asarray(audio_16k_mono, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1).astype(np.float32, copy=False)

        bs = int(beam_size) if beam_size is not None else int(self.beam_size)
        if bs < 1:
            bs = 1

        kwargs: Dict[str, Any] = {
            "beam_size": bs,
            "vad_filter": False,  # VAD is done upstream
            "condition_on_previous_text": bool(self.condition_on_previous_text),
        }

        # IMPORTANT: language can be None -> auto detect
        if self.language is not None:
            kwargs["language"] = self.language

        if self.temperature is not None:
            kwargs["temperature"] = float(self.temperature)

        if self.initial_prompt:
            kwargs["initial_prompt"] = str(self.initial_prompt)

        segments, info = self._model.transcribe(x, **kwargs)

        text_parts = []
        for s in segments:
            t = getattr(s, "text", "")
            if t:
                text_parts.append(t)

        text = "".join(text_parts).strip()
        out: Dict[str, Any] = {"text": text, "beam_size": bs}

        try:
            out["language"] = getattr(info, "language", None)
            out["language_probability"] = getattr(info, "language_probability", None)
        except Exception:
            pass

        return out
