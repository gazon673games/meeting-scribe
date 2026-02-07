# asr/diar_backend_pyannote.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

@dataclass
class DiarSegment:
    t0: float
    t1: float
    speaker: str

class PyannoteDiarizer:
    def __init__(self, device: str = "cuda"):
        # ВНИМАНИЕ: pyannote часто требует модели с HuggingFace токеном.
        # Если токен/онлайн нельзя — лучше NeMo (см. ниже).
        from pyannote.audio import Pipeline  # type: ignore
        import torch

        self._device = torch.device("cuda" if device == "cuda" else "cpu")
        # пример: конкретный pipeline зависит от того, что ты ставишь
        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        self._pipeline.to(self._device)

    def diarize(self, audio_16k: np.ndarray, t_offset: float = 0.0) -> List[DiarSegment]:
        """
        audio_16k: mono float32 16kHz
        t_offset: смещение времени этого чанка в общей шкале времени (сек)
        """
        import torch

        x = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        wav = torch.from_numpy(x).unsqueeze(0)  # (1, n)
        sample_rate = 16000

        diar = self._pipeline({"waveform": wav, "sample_rate": sample_rate})
        out: List[DiarSegment] = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            out.append(DiarSegment(t0=t_offset + float(turn.start),
                                   t1=t_offset + float(turn.end),
                                   speaker=str(speaker)))
        return out
