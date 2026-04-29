from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from diarization.application.ports import OnlineDiarizerPort


@dataclass
class _Cluster:
    label: str
    centroid: np.ndarray  # (d,)
    n: int
    last_ts: float


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = float(np.linalg.norm(x) + 1e-12)
    return x / n


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class _ResemblyzerBackend:
    """
    Lazy import wrapper. If resemblyzer isn't installed, we'll raise.
    """

    def __init__(self) -> None:
        try:
            from resemblyzer import VoiceEncoder  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Speaker diarization requires 'resemblyzer'. Install it:\n"
                "  pip install resemblyzer torch\n"
                f"Import error: {type(e).__name__}: {e}"
            )
        self._enc = VoiceEncoder()

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        # resemblyzer expects 16k mono float waveform (np.float32)
        x = np.asarray(audio_16k, dtype=np.float32)
        if x.ndim != 1:
            x = x.reshape(-1).astype(np.float32, copy=False)
        emb = self._enc.embed_utterance(x)
        return np.asarray(emb, dtype=np.float32)


class _NeMoBackend:
    """
    NeMo TiTaNet embeddings backend.
    Requires: diarization/diar_backend_nemo.py
    """

    def __init__(self, device: str = "cuda", temp_dir: Optional[Any] = None) -> None:
        from diarization.infrastructure.diar_backend_nemo import NeMoTitaNetEmbedder

        temp_path = Path(temp_dir) if temp_dir is not None else None
        self._emb = NeMoTitaNetEmbedder(device=device, temp_dir=temp_path)

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        return self._emb.embed_16k(audio_16k, sample_rate=16000)


class _SherpaOnnxBackend:
    def __init__(self, *, model_path: str, provider: str = "cpu", num_threads: int = 1) -> None:
        from diarization.infrastructure.diar_backend_sherpa import SherpaOnnxSpeakerEmbeddingBackend

        self._emb = SherpaOnnxSpeakerEmbeddingBackend(
            model_path=model_path,
            provider=provider,
            num_threads=num_threads,
        )

    def embed(self, audio_16k: np.ndarray) -> np.ndarray:
        return self._emb.embed(audio_16k)


@dataclass
class OnlineDiarizer(OnlineDiarizerPort):
    """
    Online speaker clustering (MVP):
      - each segment -> embedding
      - assign to nearest centroid if cosine similarity >= threshold
      - else create a new cluster
      - periodic pruning by age is supported (optional)

    Notes:
      - Works best when segments are >= 1.5-2.0s of clean speech.
      - Overlap (two speakers at once) will degrade quality.
    """

    similarity_threshold: float = 0.74  # tune ~0.70..0.80
    max_speakers: int = 8
    min_segment_s: float = 1.0
    prune_after_s: float = 300.0  # drop clusters not seen for 5 min (optional)
    window_s: float = 120.0  # window for estimating n_speakers

    # backend selection
    backend: str = "resemblyzer"  # "resemblyzer" | "nemo" | "sherpa_onnx"
    device: str = "cuda"
    temp_dir: Optional[Any] = None
    sherpa_model_path: str = ""
    sherpa_provider: str = "cpu"
    sherpa_num_threads: int = 1

    def __post_init__(self) -> None:
        self._backend: Optional[object] = None
        self._clusters: List[_Cluster] = []
        self._events: List[Tuple[float, str]] = []  # (ts, label) for window estimate
        self._last_error: Optional[str] = None  # NEW: store last embed/init error text

    def last_error(self) -> Optional[str]:
        return self._last_error

    def _lazy_backend(self):
        if self._backend is not None:
            return self._backend

        b = (self.backend or "resemblyzer").strip().lower()
        if b == "nemo":
            self._backend = _NeMoBackend(device=self.device, temp_dir=self.temp_dir)
        elif b == "sherpa_onnx":
            self._backend = _SherpaOnnxBackend(
                model_path=self.sherpa_model_path,
                provider=self.sherpa_provider,
                num_threads=self.sherpa_num_threads,
            )
        else:
            self._backend = _ResemblyzerBackend()
        return self._backend

    def _next_label(self) -> str:
        return f"S{len(self._clusters) + 1}"

    def _prune(self, now: float) -> None:
        if self.prune_after_s <= 0:
            return
        keep: List[_Cluster] = []
        for c in self._clusters:
            if (now - c.last_ts) <= self.prune_after_s:
                keep.append(c)
        self._clusters = keep

        if self.window_s > 0:
            cutoff = now - self.window_s
            self._events = [(t, l) for (t, l) in self._events if t >= cutoff]

    def assign(self, audio_16k: np.ndarray, ts: Optional[float] = None) -> Tuple[str, Optional[int]]:
        """Returns: (speaker_label, n_speakers_estimate_or_None)"""
        label, n_speakers, _, _ = self.assign_with_debug(audio_16k, ts)
        return (label, n_speakers)

    def estimate_n_speakers(self, now: Optional[float] = None) -> Optional[int]:
        if self.window_s <= 0:
            return None
        t = float(now if now is not None else time.time())
        cutoff = t - float(self.window_s)
        labels = {l for (ts, l) in self._events if ts >= cutoff and l != "S?"}
        if not labels:
            return None
        return int(len(labels))

    def assign_with_debug(
        self, audio_16k: np.ndarray, ts: Optional[float] = None
    ) -> tuple[str, Optional[int], float, bool]:
        """
        Returns: (label, n_speakers, best_sim, created_new_cluster)
        """
        now = float(ts if ts is not None else time.time())
        dur_s = float(np.asarray(audio_16k).shape[0]) / 16000.0
        if dur_s < float(self.min_segment_s):
            self._prune(now)
            return ("S?", self.estimate_n_speakers(now), -1.0, False)

        try:
            emb = self._lazy_backend().embed(audio_16k)  # type: ignore[attr-defined]
            self._last_error = None
        except Exception as e:
            self._last_error = f"{type(e).__name__}: {e}"
            self._prune(now)
            return (f"S_ERR:{type(e).__name__}", self.estimate_n_speakers(now), -1.0, False)

        emb = _l2norm(np.asarray(emb, dtype=np.float32))
        self._prune(now)

        if not self._clusters:
            label = self._next_label()
            self._clusters.append(_Cluster(label=label, centroid=emb, n=1, last_ts=now))
            self._events.append((now, label))
            return (label, self.estimate_n_speakers(now), 1.0, True)

        best_i = -1
        best_sim = -1.0
        for i, c in enumerate(self._clusters):
            sim = _cos_sim(emb, c.centroid)
            if sim > best_sim:
                best_sim = sim
                best_i = i

        created = False
        if best_i >= 0 and best_sim >= float(self.similarity_threshold):
            c = self._clusters[best_i]
            c.centroid = _l2norm((c.centroid * float(c.n) + emb) / float(c.n + 1))
            c.n += 1
            c.last_ts = now
            label = c.label
        else:
            if len(self._clusters) < int(self.max_speakers):
                label = self._next_label()
                self._clusters.append(_Cluster(label=label, centroid=emb, n=1, last_ts=now))
                created = True
            else:
                label = self._clusters[best_i].label if best_i >= 0 else "S?"

        self._events.append((now, label))
        return (label, self.estimate_n_speakers(now), float(best_sim), created)
