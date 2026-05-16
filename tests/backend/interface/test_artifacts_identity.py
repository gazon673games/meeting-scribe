from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from interface.session_controller_parts.artifacts_identity import (
    IdentityProcessor,
    create_identity_store,
    identity_snapshot_from_runtime,
    snapshot_embedding_model,
    speech_quality_score,
)


class _FakeStore:
    def __init__(self, match_status: str = "new", person_id: str = "", similarity: float = 0.0) -> None:
        self.match_status = match_status
        self.person_id = person_id
        self.similarity = similarity
        self.saved = []
        self.links = []
        self.updated = []
        self.created = []

    def save_embedding(self, session_id, session_speaker_id, embedding, metadata):  # noqa: ANN001
        self.saved.append((session_id, session_speaker_id, embedding.copy(), dict(metadata)))
        return f"emb-{session_speaker_id}"

    def find_best_match(self, *, embedding, embedding_model, threshold_auto, threshold_uncertain):  # noqa: ANN001
        self.last_match_args = {
            "embedding": embedding,
            "embedding_model": embedding_model,
            "threshold_auto": threshold_auto,
            "threshold_uncertain": threshold_uncertain,
        }
        return SimpleNamespace(match_status=self.match_status, person_id=self.person_id, match_similarity=self.similarity)

    def link_session_speaker(self, *args):  # noqa: ANN002
        self.links.append(args)

    def update_profile_centroid(self, person_id):  # noqa: ANN001
        self.updated.append(person_id)

    def create_profile(self, embedding_id, metadata):  # noqa: ANN001
        self.created.append((embedding_id, dict(metadata)))
        return "person-new"


def _settings(**overrides):  # noqa: ANN003
    values = {
        "speaker_identity_enabled": True,
        "speaker_identity_persistent_profiles_enabled": True,
        "speaker_identity_backend": "file",
        "speaker_identity_auto_match_threshold": 0.84,
        "speaker_identity_uncertain_match_threshold": 0.74,
        "speaker_identity_min_speech_ms_for_embedding": 1000,
        "speaker_identity_min_quality_score": 0.5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _snapshot():
    return {
        "mic": {
            "S1": {
                "embedding": np.asarray([1.0, 0.0], dtype=np.float32),
                "embedding_model": "speaker-model",
                "embedding_dim": 2,
            }
        }
    }


class ArtifactsIdentityTests(unittest.TestCase):
    def test_identity_snapshot_helpers_are_defensive(self) -> None:
        self.assertEqual(identity_snapshot_from_runtime(object()), {})
        self.assertEqual(identity_snapshot_from_runtime(SimpleNamespace(identity_snapshot=lambda: "bad")), {})
        self.assertEqual(identity_snapshot_from_runtime(SimpleNamespace(identity_snapshot=lambda: (_ for _ in ()).throw(RuntimeError("boom")))), {})
        snapshot = identity_snapshot_from_runtime(SimpleNamespace(identity_snapshot=_snapshot))
        self.assertEqual(snapshot["mic"]["S1"]["embedding_model"], "speaker-model")
        np.testing.assert_array_equal(snapshot["mic"]["S1"]["embedding"], np.asarray([1.0, 0.0], dtype=np.float32))
        self.assertEqual(snapshot_embedding_model({"mic": {"S1": {"embedding_model": "m1"}}}), "m1")
        self.assertEqual(snapshot_embedding_model({"mic": {"S1": {}}}), "")
        self.assertEqual(speech_quality_score(total_speech_ms=250, min_speech_ms=1000), 0.25)
        self.assertEqual(speech_quality_score(total_speech_ms=250, min_speech_ms=0), 1.0)

    def test_create_identity_store_respects_persistence_and_backend(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.assertIsNone(create_identity_store(app_data_dir=root, settings=_settings(), persistent_enabled=False))
            self.assertIsNone(create_identity_store(app_data_dir=root, settings=_settings(speaker_identity_backend="remote"), persistent_enabled=True))
            self.assertIsNotNone(create_identity_store(app_data_dir=root, settings=_settings(), persistent_enabled=True))

    def test_processor_applies_probable_uncertain_and_new_matches_to_speakers_and_lines(self) -> None:
        cases = [
            ("probable", "person-existing", 0.93, "accepted_auto"),
            ("uncertain", "person-maybe", 0.78, "uncertain"),
            ("new", "", 0.0, "created_new"),
        ]
        for status, person_id, similarity, link_status in cases:
            store = _FakeStore(match_status=status, person_id=person_id, similarity=similarity)
            processor = IdentityProcessor(
                session_id="session-1",
                settings=_settings(),
                identity_enabled=True,
                store=store,
                snapshot=_snapshot(),
                embedding_model="speaker-model",
            )
            speaker = {"session_speaker_id": "spk1", "stream": "mic", "label": "S1", "total_speech_ms": 1200}

            processor.apply_to_speakers([speaker])
            lines = [{"session_speaker_id": "spk1"}]
            processor.apply_to_lines(lines, [speaker])

            expected_person = person_id or "person-new"
            self.assertEqual(speaker["embedding_id"], "emb-spk1")
            self.assertEqual(speaker["matched_person_id"], expected_person)
            self.assertEqual(speaker["match_status"], status)
            self.assertEqual(store.links[0][-1], link_status)
            self.assertEqual(lines[0]["person_id"], expected_person)
            self.assertIsInstance(lines[0]["identity_confidence"], float)

    def test_processor_rejects_low_quality_or_missing_embeddings_without_store_calls(self) -> None:
        store = _FakeStore()
        low_quality = {"session_speaker_id": "spk1", "stream": "mic", "label": "S1", "total_speech_ms": 100}
        processor = IdentityProcessor(
            session_id="session-1",
            settings=_settings(),
            identity_enabled=True,
            store=store,
            snapshot=_snapshot(),
            embedding_model="speaker-model",
        )
        processor.apply_to_speakers([low_quality])
        self.assertEqual(low_quality["match_status"], "rejected")
        self.assertEqual(store.saved, [])

        no_identity = {"session_speaker_id": "spk2", "stream": "mic", "label": "S2", "total_speech_ms": 1200}
        processor = IdentityProcessor(
            session_id="session-1",
            settings=_settings(),
            identity_enabled=False,
            store=store,
            snapshot={},
            embedding_model="",
        )
        processor.apply_to_speakers([no_identity])
        self.assertEqual(no_identity["match_status"], "new")
        self.assertIsNone(no_identity["matched_person_id"])


if __name__ == "__main__":
    unittest.main()
