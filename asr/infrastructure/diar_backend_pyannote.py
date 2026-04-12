from __future__ import annotations

from typing import List

import numpy as np

from asr.domain.segments import DiarSegment


class PyannoteDiarizer:
    def __init__(self, device: str = "cuda") -> None:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "pyannote.audio is not installed or failed to import.\n"
                "Install example:\n"
                "  pip install pyannote.audio\n"
                f"Import error: {type(e).__name__}: {e}"
            )

        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "pyannote.audio requires torch.\n"
                "Install torch compatible with your CUDA.\n"
                f"Import error: {type(e).__name__}: {e}"
            )

        self._torch = torch
        self._device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self._pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        try:
            self._pipeline.to(self._device)
        except Exception:
            pass

    def diarize(self, audio_16k: np.ndarray, *, t_offset: float = 0.0) -> List[DiarSegment]:
        x = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return []

        wav = self._torch.from_numpy(x).unsqueeze(0)
        diar = self._pipeline({"waveform": wav, "sample_rate": 16000})

        out: List[DiarSegment] = []
        for turn, _, label in diar.itertracks(yield_label=True):
            out.append(
                DiarSegment(
                    t0=float(t_offset) + float(turn.start),
                    t1=float(t_offset) + float(turn.end),
                    speaker=str(label),
                )
            )
        return out
