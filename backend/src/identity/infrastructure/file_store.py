from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from identity.application.store import MatchResult, SpeakerProfile


class FileSpeakerIdentityStore:
    def __init__(self, app_data_dir: Path | str) -> None:
        self._app_data_dir = Path(app_data_dir).expanduser().resolve()
        self._identity_dir = self._app_data_dir / "identity"
        self._embeddings_dir = self._identity_dir / "embeddings"
        self._profiles_path = self._identity_dir / "profiles.jsonl"
        self._links_path = self._identity_dir / "links.jsonl"

        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        self._identity_dir.mkdir(parents=True, exist_ok=True)

    def save_embedding(
        self,
        session_id: str,
        session_speaker_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any],
    ) -> str:
        vector = _normalized_vector(embedding)
        embedding_id = _make_id("emb")
        path = self._embedding_path(embedding_id)
        np.save(path, vector.astype(np.float32, copy=False))
        record = {
            "type": "embedding_saved",
            "embedding_id": embedding_id,
            "session_id": str(session_id),
            "session_speaker_id": str(session_speaker_id),
            "embedding_model": str(metadata.get("embedding_model") or ""),
            "embedding_dim": int(vector.shape[0]),
            "metadata": dict(metadata or {}),
            "ts": time.time(),
        }
        self._append_jsonl(self._links_path, record)
        return embedding_id

    def load_embedding(self, embedding_id: str) -> np.ndarray:
        path = self._embedding_path(embedding_id)
        if not path.exists():
            raise FileNotFoundError(f"Embedding does not exist: {embedding_id}")
        raw = np.load(path)
        return _normalized_vector(raw)

    def list_profiles(self) -> List[SpeakerProfile]:
        latest = self._latest_profile_records()
        profiles: List[SpeakerProfile] = []
        for record in latest.values():
            profiles.append(
                SpeakerProfile(
                    person_id=str(record.get("person_id") or ""),
                    embedding_ids=[str(item) for item in record.get("embedding_ids", []) if str(item).strip()],
                    centroid_embedding_id=str(record.get("centroid_embedding_id") or ""),
                    embedding_model=str(record.get("embedding_model") or ""),
                    embedding_dim=int(record.get("embedding_dim") or 0),
                    metadata=dict(record.get("metadata") or {}),
                    created_at=float(record.get("created_at") or 0.0),
                    updated_at=float(record.get("updated_at") or 0.0),
                )
            )
        return profiles

    def find_best_match(
        self,
        embedding: np.ndarray,
        embedding_model: str,
        threshold_auto: float,
        threshold_uncertain: float,
    ) -> MatchResult:
        vector = _normalized_vector(embedding)
        model = str(embedding_model or "").strip()
        if not model:
            return MatchResult(person_id=None, match_similarity=0.0, match_status="new")
        best_person_id = None
        best_score = -1.0

        for profile in self.list_profiles():
            profile_model = str(profile.embedding_model or "").strip()
            if not profile_model or profile_model != model:
                continue
            centroid_id = str(profile.centroid_embedding_id or "")
            if not centroid_id:
                continue
            try:
                centroid = self.load_embedding(centroid_id)
            except Exception:
                continue
            if centroid.shape != vector.shape:
                continue
            score = _cosine_similarity(vector, centroid)
            if score > best_score:
                best_score = score
                best_person_id = profile.person_id

        if best_person_id is None:
            return MatchResult(person_id=None, match_similarity=0.0, match_status="new")
        if best_score >= float(threshold_auto):
            return MatchResult(person_id=best_person_id, match_similarity=float(best_score), match_status="probable")
        if best_score >= float(threshold_uncertain):
            return MatchResult(person_id=best_person_id, match_similarity=float(best_score), match_status="uncertain")
        return MatchResult(person_id=None, match_similarity=float(best_score), match_status="new")

    def create_profile(self, embedding_id: str, metadata: Dict[str, Any]) -> str:
        now = time.time()
        person_id = _make_id("person")
        record = {
            "person_id": person_id,
            "embedding_ids": [str(embedding_id)],
            "centroid_embedding_id": str(embedding_id),
            "embedding_model": str(metadata.get("embedding_model") or ""),
            "embedding_dim": int(metadata.get("embedding_dim") or 0),
            "metadata": dict(metadata or {}),
            "created_at": now,
            "updated_at": now,
            "ts": now,
        }
        self._append_jsonl(self._profiles_path, record)
        return person_id

    def link_session_speaker(
        self,
        session_id: str,
        session_speaker_id: str,
        person_id: str,
        embedding_id: str,
        similarity: float,
        decision: str,
    ) -> None:
        link = {
            "type": "speaker_link",
            "session_id": str(session_id),
            "session_speaker_id": str(session_speaker_id),
            "person_id": str(person_id),
            "embedding_id": str(embedding_id),
            "similarity": float(similarity),
            "decision": str(decision),
            "ts": time.time(),
        }
        self._append_jsonl(self._links_path, link)

        profile = self._latest_profile_records().get(str(person_id))
        if profile is None:
            return
        embedding_ids = [str(item) for item in profile.get("embedding_ids", []) if str(item).strip()]
        if embedding_id and embedding_id not in embedding_ids:
            embedding_ids.append(str(embedding_id))
        profile["embedding_ids"] = embedding_ids
        profile["updated_at"] = time.time()
        profile["ts"] = time.time()
        self._append_jsonl(self._profiles_path, profile)

    def update_profile_centroid(self, person_id: str) -> None:
        profiles = self._latest_profile_records()
        record = profiles.get(str(person_id))
        if record is None:
            return
        embedding_ids = [str(item) for item in record.get("embedding_ids", []) if str(item).strip()]
        vectors: List[np.ndarray] = []
        for embedding_id in embedding_ids:
            try:
                vectors.append(self.load_embedding(embedding_id))
            except Exception:
                continue
        if not vectors:
            return
        centroid = _normalized_vector(np.mean(np.stack(vectors, axis=0), axis=0))
        centroid_id = _make_id("centroid")
        np.save(self._embedding_path(centroid_id), centroid.astype(np.float32, copy=False))

        updated = dict(record)
        updated["centroid_embedding_id"] = centroid_id
        updated["embedding_dim"] = int(centroid.shape[0])
        updated["updated_at"] = time.time()
        updated["ts"] = time.time()
        self._append_jsonl(self._profiles_path, updated)

    def _latest_profile_records(self) -> Dict[str, Dict[str, Any]]:
        latest: Dict[str, Dict[str, Any]] = {}
        for record in self._iter_jsonl(self._profiles_path):
            person_id = str(record.get("person_id") or "").strip()
            if not person_id:
                continue
            latest[person_id] = dict(record)
        return latest

    def _embedding_path(self, embedding_id: str) -> Path:
        safe_id = Path(str(embedding_id or "")).name
        if not safe_id:
            raise ValueError("embedding_id is required")
        return self._embeddings_dir / f"{safe_id}.npy"

    @staticmethod
    def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    out.append(item)
        return out


def _normalized_vector(raw: np.ndarray) -> np.ndarray:
    vector = np.asarray(raw, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32, copy=False)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = _normalized_vector(left)
    right_norm = _normalized_vector(right)
    return float(np.dot(left_norm, right_norm))


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"
