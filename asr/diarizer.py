# --- File: D:\work\own\voice2textTest\asr\diarizer.py ---
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


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
        # VoiceEncoder.embed_utterance returns (d,)
        emb = self._enc.embed_utterance(x)
        return np.asarray(emb, dtype=np.float32)


@dataclass
class OnlineDiarizer:
    """
    Online speaker clustering (MVP):
      - each segment -> embedding
      - assign to nearest centroid if cosine similarity >= threshold
      - else create a new cluster
      - periodic pruning by age is supported (optional)

    Notes:
      - Works best when segments are >= 1s of clean speech.
      - Overlap (two speakers at once) will degrade quality.
    """
    similarity_threshold: float = 0.74  # tune 0.70..0.80
    max_speakers: int = 8
    min_segment_s: float = 1.0
    prune_after_s: float = 300.0  # drop clusters not seen for 5 min (optional)
    window_s: float = 120.0       # window for estimating n_speakers

    def __post_init__(self) -> None:
        self._backend: Optional[_ResemblyzerBackend] = None
        self._clusters: List[_Cluster] = []
        self._events: List[Tuple[float, str]] = []  # (ts, label) for window estimate

    def _lazy_backend(self) -> _ResemblyzerBackend:
        if self._backend is None:
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

        # also prune events buffer
        if self.window_s > 0:
            cutoff = now - self.window_s
            self._events = [(t, l) for (t, l) in self._events if t >= cutoff]

    def assign(self, audio_16k: np.ndarray, ts: Optional[float] = None) -> Tuple[str, Optional[int]]:
        """
        Returns:
          (speaker_label, n_speakers_estimate_or_None)
        """
        now = float(ts if ts is not None else time.time())

        # segment length guard
        dur_s = float(np.asarray(audio_16k).shape[0]) / 16000.0
        if dur_s < float(self.min_segment_s):
            # Not enough info; don't create clusters. Still update estimate window as unknown.
            self._prune(now)
            return ("S?", self.estimate_n_speakers(now))

        # embedding
        try:
            emb = self._lazy_backend().embed(audio_16k)
        except Exception:
            self._prune(now)
            return ("S?", self.estimate_n_speakers(now))

        emb = _l2norm(emb)

        self._prune(now)

        # no clusters yet
        if not self._clusters:
            label = self._next_label()
            self._clusters.append(_Cluster(label=label, centroid=emb, n=1, last_ts=now))
            self._events.append((now, label))
            return (label, self.estimate_n_speakers(now))

        # find best cluster
        best_i = -1
        best_sim = -1.0
        for i, c in enumerate(self._clusters):
            sim = _cos_sim(emb, c.centroid)
            if sim > best_sim:
                best_sim = sim
                best_i = i

        if best_i >= 0 and best_sim >= float(self.similarity_threshold):
            c = self._clusters[best_i]
            # centroid update (running mean)
            new_centroid = _l2norm((c.centroid * float(c.n) + emb) / float(c.n + 1))
            c.centroid = new_centroid
            c.n += 1
            c.last_ts = now
            label = c.label
        else:
            if len(self._clusters) < int(self.max_speakers):
                label = self._next_label()
                self._clusters.append(_Cluster(label=label, centroid=emb, n=1, last_ts=now))
            else:
                # fallback to best even if below threshold
                label = self._clusters[best_i].label if best_i >= 0 else "S?"

        self._events.append((now, label))
        return (label, self.estimate_n_speakers(now))

    def estimate_n_speakers(self, now: Optional[float] = None) -> Optional[int]:
        """
        Estimate number of distinct speakers seen in recent window.
        """
        if self.window_s <= 0:
            return None
        t = float(now if now is not None else time.time())
        cutoff = t - float(self.window_s)
        labels = {l for (ts, l) in self._events if ts >= cutoff and l != "S?"}
        if not labels:
            return None
        return int(len(labels))
