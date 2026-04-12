from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _to_float32_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x


@dataclass
class NeMoTitaNetEmbedder:
    """
    NeMo speaker embeddings via pretrained TiTaNet.

    Implementation notes:
      - For Windows robustness, we use "file path" inference:
        write segment to a temp wav, then ask NeMo to embed that file.
      - This avoids API drift around in-memory waveform embedding.
    """
    device: str = "cuda"
    model_name: str = "titanet_large"  # common pretrained speaker embedding model

    def __post_init__(self) -> None:
        self._model = None
        self._torch = None

        # soundfile is required for temp wav writing
        try:
            import soundfile as sf  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "NeMo diarization backend requires 'soundfile'. Install:\n"
                "  pip install soundfile\n"
                f"Import error: {type(e).__name__}: {e}"
            )
        self._sf = sf

    def _lazy_init(self) -> None:
        if self._model is not None:
            return

        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "NeMo requires torch. Install torch with CUDA.\n"
                f"Import error: {type(e).__name__}: {e}"
            )

        try:
            from nemo.collections.asr.models import EncDecSpeakerLabelModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Failed to import NeMo speaker model.\n"
                "Install: pip install nemo_toolkit[asr]\n"
                f"Import error: {type(e).__name__}: {e}"
            )

        self._torch = torch
        dev = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"

        # downloads weights from NVIDIA NGC via NeMo (no HF token)
        model = EncDecSpeakerLabelModel.from_pretrained(model_name=self.model_name)
        model = model.to(dev)
        model.eval()

        self._model = model
        self.device = dev  # normalize

    def embed_16k(self, audio_16k: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        audio_16k: mono float32 waveform @16k
        returns: (d,) float32
        """
        self._lazy_init()

        x = _to_float32_mono(audio_16k)
        if x.size < int(0.8 * sample_rate):
            # too short => unstable embedding
            raise RuntimeError("segment too short for NeMo embedding (<0.8s)")

        # Write temp wav (PCM_16 for compatibility)
        fd = None
        path = None
        try:
            fd, path = tempfile.mkstemp(prefix="nemo_seg_", suffix=".wav")
            os.close(fd)
            fd = None

            self._sf.write(path, x, samplerate=sample_rate, subtype="PCM_16")

            # NeMo speaker model most-stable API: file paths
            # Returns embeddings for each file.
            embs = self._model.get_embedding(audio_file_paths=[path])
            # embs: torch.Tensor [B, D]
            emb = embs[0].detach().cpu().numpy().astype(np.float32, copy=False)
            return emb

        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            if path is not None:
                try:
                    os.remove(path)
                except Exception:
                    pass
