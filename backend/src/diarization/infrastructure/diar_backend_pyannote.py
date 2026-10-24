from __future__ import annotations

import logging

import numpy as np

from diarization.application.ports import PyannoteDiarizerPort
from diarization.domain.segments import DiarSegment

_logger = logging.getLogger(__name__)


class PyannoteDiarizer(PyannoteDiarizerPort):
    def __init__(self, device: str = 'cuda') -> None:
        try:
            from pyannote.audio import Pipeline  # type: ignore
        except Exception as error:
            raise RuntimeError(
                'pyannote.audio is not installed or failed to import.\n'
                'Install example:\n'
                '  pip install pyannote.audio\n'
                f'Import error: {type(error).__name__}: {error}'
            )

        try:
            import torch  # type: ignore
        except Exception as exception:
            raise RuntimeError(
                'pyannote.audio requires torch.\n'
                'Install torch compatible with your CUDA.\n'
                f'Import error: {type(exception).__name__}: {exception}'
            )

        self._torch = torch
        self._device = torch.device(
            'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        )
        self._pipeline = Pipeline.from_pretrained(
            'pyannote/speaker-diarization'
        )
        try:
            self._pipeline.to(self._device)
        except Exception:
            _logger.debug(
                'Best-effort operation failed in __init__', exc_info=True
            )

    def diarize(
        self, audio_16k: np.ndarray, *, t_offset: float = 0.0
    ) -> list[DiarSegment]:
        x = np.asarray(audio_16k, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return []

        wav = self._torch.from_numpy(x).unsqueeze(0)
        diar = self._pipeline({'waveform': wav, 'sample_rate': 16000})

        out: list[DiarSegment] = []
        for turn, _, label in diar.itertracks(yield_label=True):
            out.append(
                DiarSegment(
                    t0=float(t_offset) + float(turn.start),
                    t1=float(t_offset) + float(turn.end),
                    speaker=str(label),
                )
            )
        return out
