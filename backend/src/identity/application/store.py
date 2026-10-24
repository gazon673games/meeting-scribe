from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

MatchStatus = Literal['new', 'probable', 'confirmed', 'uncertain', 'rejected']


@dataclass(frozen=True)
class SpeakerProfile:
    person_id: str
    embedding_ids: list[str] = field(default_factory=list)
    centroid_embedding_id: str = ''
    embedding_model: str = ''
    embedding_dim: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(frozen=True)
class MatchResult:
    person_id: str | None
    match_similarity: float
    match_status: MatchStatus


class SpeakerIdentityStore(Protocol):
    def save_embedding(
        self,
        session_id: str,
        session_speaker_id: str,
        embedding: np.ndarray,
        metadata: dict[str, Any],
    ) -> str: ...

    def load_embedding(self, embedding_id: str) -> np.ndarray: ...

    def list_profiles(self) -> list[SpeakerProfile]: ...

    def find_best_match(
        self,
        embedding: np.ndarray,
        embedding_model: str,
        threshold_auto: float,
        threshold_uncertain: float,
    ) -> MatchResult: ...

    def create_profile(
        self, embedding_id: str, metadata: dict[str, Any]
    ) -> str: ...

    def link_session_speaker(
        self,
        session_id: str,
        session_speaker_id: str,
        person_id: str,
        embedding_id: str,
        similarity: float,
        decision: str,
    ) -> None: ...

    def update_profile_centroid(self, person_id: str) -> None: ...
