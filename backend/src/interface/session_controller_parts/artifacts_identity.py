from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from application.asr_session import ASRSessionSettings
from identity.application.store import SpeakerIdentityStore
from identity.infrastructure.file_store import FileSpeakerIdentityStore


def identity_snapshot_from_runtime(asr: Any) -> Dict[str, Dict[str, dict]]:
    export = getattr(asr, "identity_snapshot", None)
    if not callable(export):
        return {}
    try:
        raw = export()
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def snapshot_embedding_model(snapshot: Dict[str, Dict[str, dict]]) -> str:
    for stream_data in snapshot.values():
        if not isinstance(stream_data, dict):
            continue
        for payload in stream_data.values():
            if not isinstance(payload, dict):
                continue
            model_name = str(payload.get("embedding_model") or "").strip()
            if model_name:
                return model_name
    return ""


def speech_quality_score(*, total_speech_ms: int, min_speech_ms: int) -> float:
    if min_speech_ms <= 0:
        return 1.0
    return max(0.0, min(1.0, float(total_speech_ms) / float(min_speech_ms)))


def create_identity_store(
    *,
    app_data_dir: Path,
    settings: Optional[ASRSessionSettings],
    persistent_enabled: bool,
) -> Optional[SpeakerIdentityStore]:
    if not persistent_enabled:
        return None
    backend_name = str(settings.speaker_identity_backend if settings else "file").strip().lower()
    if backend_name != "file":
        return None
    return FileSpeakerIdentityStore(app_data_dir)


@dataclass
class IdentityProcessor:
    session_id: str
    settings: Optional[ASRSessionSettings]
    identity_enabled: bool
    store: Optional[SpeakerIdentityStore]
    snapshot: Dict[str, Dict[str, dict]]
    embedding_model: str

    @classmethod
    def from_runtime(
        cls,
        *,
        session_id: str,
        asr: Any,
        settings: Optional[ASRSessionSettings],
        app_data_dir: Path,
    ) -> "IdentityProcessor":
        identity_enabled = bool(settings and settings.speaker_identity_enabled)
        persistent_enabled = bool(settings and settings.speaker_identity_persistent_profiles_enabled and identity_enabled)
        snapshot = identity_snapshot_from_runtime(asr) if identity_enabled else {}
        return cls(
            session_id=str(session_id),
            settings=settings,
            identity_enabled=identity_enabled,
            store=create_identity_store(
                app_data_dir=app_data_dir,
                settings=settings,
                persistent_enabled=persistent_enabled,
            ),
            snapshot=snapshot,
            embedding_model=snapshot_embedding_model(snapshot),
        )

    def apply_to_speakers(self, speakers: list[Dict[str, Any]]) -> None:
        for speaker in speakers:
            self._apply_to_speaker(speaker)

    def apply_to_lines(self, lines: list[Dict[str, Any]], speakers: list[Dict[str, Any]]) -> None:
        speakers_by_id = {str(item.get("session_speaker_id") or ""): item for item in speakers}
        for line in lines:
            speaker = speakers_by_id.get(str(line.get("session_speaker_id") or ""), {})
            line["person_id"] = speaker.get("matched_person_id")
            confidence = speaker.get("match_similarity")
            line["identity_confidence"] = float(confidence) if speaker.get("matched_person_id") else None

    def _apply_to_speaker(self, speaker: Dict[str, Any]) -> None:
        self._set_identity_defaults(speaker)
        if not self.identity_enabled:
            return

        quality_score = self._speaker_quality_score(speaker)
        if quality_score < self._min_quality():
            speaker["match_status"] = "rejected"
            return

        embed_info = self._embed_info(speaker)
        embedding = embed_info.get("embedding") if isinstance(embed_info, dict) else None
        model_name = str(embed_info.get("embedding_model") or "") if isinstance(embed_info, dict) else ""
        if not isinstance(embedding, np.ndarray) or self.store is None:
            return

        metadata = self._embedding_metadata(speaker, embedding, embed_info, quality_score)
        session_speaker_id = str(speaker.get("session_speaker_id") or "")
        embedding_id = self.store.save_embedding(self.session_id, session_speaker_id, embedding, metadata)
        speaker["embedding_id"] = embedding_id

        match = self.store.find_best_match(
            embedding=embedding,
            embedding_model=model_name,
            threshold_auto=self._auto_threshold(),
            threshold_uncertain=self._uncertain_threshold(),
        )
        self._apply_match_decision(
            speaker=speaker,
            session_speaker_id=session_speaker_id,
            embedding_id=embedding_id,
            match=match,
            metadata=metadata,
        )

    def _apply_match_decision(
        self,
        *,
        speaker: Dict[str, Any],
        session_speaker_id: str,
        embedding_id: str,
        match: Any,
        metadata: Dict[str, Any],
    ) -> None:
        assert self.store is not None
        if match.match_status == "probable" and match.person_id:
            person_id = str(match.person_id)
            similarity = float(match.match_similarity)
            speaker["matched_person_id"] = person_id
            speaker["match_similarity"] = similarity
            speaker["match_status"] = "probable"
            self.store.link_session_speaker(
                self.session_id,
                session_speaker_id,
                person_id,
                embedding_id,
                similarity,
                "accepted_auto",
            )
            self.store.update_profile_centroid(person_id)
            return

        if match.match_status == "uncertain" and match.person_id:
            person_id = str(match.person_id)
            similarity = float(match.match_similarity)
            speaker["matched_person_id"] = person_id
            speaker["match_similarity"] = similarity
            speaker["match_status"] = "uncertain"
            self.store.link_session_speaker(
                self.session_id,
                session_speaker_id,
                person_id,
                embedding_id,
                similarity,
                "uncertain",
            )
            return

        person_id = str(self.store.create_profile(embedding_id, metadata))
        speaker["matched_person_id"] = person_id
        speaker["match_similarity"] = 0.0
        speaker["match_status"] = "new"
        self.store.link_session_speaker(
            self.session_id,
            session_speaker_id,
            person_id,
            embedding_id,
            0.0,
            "created_new",
        )
        self.store.update_profile_centroid(person_id)

    def _set_identity_defaults(self, speaker: Dict[str, Any]) -> None:
        speaker["embedding_id"] = None
        speaker["matched_person_id"] = None
        speaker["match_similarity"] = 0.0
        speaker["match_status"] = "new"

    def _embed_info(self, speaker: Dict[str, Any]) -> Dict[str, Any]:
        key_stream = str(speaker.get("stream") or "mix")
        key_label = str(speaker.get("label") or "")
        stream_payload = self.snapshot.get(key_stream)
        if not isinstance(stream_payload, dict):
            return {}
        payload = stream_payload.get(key_label)
        return payload if isinstance(payload, dict) else {}

    def _embedding_metadata(
        self,
        speaker: Dict[str, Any],
        embedding: np.ndarray,
        embed_info: Dict[str, Any],
        quality_score: float,
    ) -> Dict[str, Any]:
        key_stream = str(speaker.get("stream") or "mix")
        key_label = str(speaker.get("label") or "")
        embedding_dim = int(embed_info.get("embedding_dim") or int(embedding.shape[0]))
        return {
            "embedding_model": str(embed_info.get("embedding_model") or ""),
            "embedding_dim": embedding_dim,
            "quality_score": float(quality_score),
            "stream": key_stream,
            "label": key_label,
        }

    def _speaker_quality_score(self, speaker: Dict[str, Any]) -> float:
        total_speech_ms = int(speaker.get("total_speech_ms") or 0)
        return speech_quality_score(
            total_speech_ms=total_speech_ms,
            min_speech_ms=self._min_speech_ms(),
        )

    def _auto_threshold(self) -> float:
        return float(self.settings.speaker_identity_auto_match_threshold if self.settings else 0.84)

    def _uncertain_threshold(self) -> float:
        return float(self.settings.speaker_identity_uncertain_match_threshold if self.settings else 0.74)

    def _min_speech_ms(self) -> int:
        return int(self.settings.speaker_identity_min_speech_ms_for_embedding if self.settings else 8000)

    def _min_quality(self) -> float:
        return float(self.settings.speaker_identity_min_quality_score if self.settings else 0.65)
