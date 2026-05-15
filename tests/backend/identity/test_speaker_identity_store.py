from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from identity.infrastructure.file_store import FileSpeakerIdentityStore


class SpeakerIdentityStoreTests(unittest.TestCase):
    def test_save_and_load_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = FileSpeakerIdentityStore(Path(raw_root))
            embedding_id = store.save_embedding(
                "sess_1",
                "spk_1",
                np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
                {"embedding_model": "resemblyzer", "embedding_dim": 3},
            )
            loaded = store.load_embedding(embedding_id)

            self.assertEqual(loaded.dtype, np.float32)
            self.assertAlmostEqual(float(np.linalg.norm(loaded)), 1.0, places=5)
            self.assertEqual(loaded.shape[0], 3)

    def test_cosine_similarity_matching_prefers_best_profile(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = FileSpeakerIdentityStore(Path(raw_root))

            emb_a = store.save_embedding("sess_1", "spk_a", np.asarray([1.0, 0.0], dtype=np.float32), {"embedding_model": "m1", "embedding_dim": 2})
            person_a = store.create_profile(emb_a, {"embedding_model": "m1", "embedding_dim": 2})

            emb_b = store.save_embedding("sess_1", "spk_b", np.asarray([0.0, 1.0], dtype=np.float32), {"embedding_model": "m1", "embedding_dim": 2})
            person_b = store.create_profile(emb_b, {"embedding_model": "m1", "embedding_dim": 2})

            query = np.asarray([0.05, 0.95], dtype=np.float32)
            match = store.find_best_match(query, "m1", threshold_auto=0.84, threshold_uncertain=0.74)

            self.assertEqual(match.person_id, person_b)
            self.assertEqual(match.match_status, "probable")
            self.assertGreater(match.match_similarity, 0.9)
            self.assertNotEqual(person_a, person_b)

    def test_new_profile_creation_and_listing(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = FileSpeakerIdentityStore(Path(raw_root))
            emb = store.save_embedding(
                "sess_1",
                "spk_1",
                np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                {"embedding_model": "m2", "embedding_dim": 3},
            )
            person_id = store.create_profile(emb, {"embedding_model": "m2", "embedding_dim": 3})
            profiles = store.list_profiles()

            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0].person_id, person_id)
            self.assertEqual(profiles[0].embedding_ids, [emb])
            self.assertEqual(profiles[0].embedding_model, "m2")

    def test_accepted_auto_match_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = FileSpeakerIdentityStore(Path(raw_root))
            emb = store.save_embedding(
                "sess_1",
                "spk_1",
                np.asarray([1.0, 0.0], dtype=np.float32),
                {"embedding_model": "m1", "embedding_dim": 2},
            )
            person_id = store.create_profile(emb, {"embedding_model": "m1", "embedding_dim": 2})
            match = store.find_best_match(
                np.asarray([0.99, 0.01], dtype=np.float32),
                "m1",
                threshold_auto=0.84,
                threshold_uncertain=0.74,
            )

            self.assertEqual(match.match_status, "probable")
            self.assertEqual(match.person_id, person_id)
            self.assertGreaterEqual(match.match_similarity, 0.84)

    def test_uncertain_match_between_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = FileSpeakerIdentityStore(Path(raw_root))
            emb = store.save_embedding(
                "sess_1",
                "spk_1",
                np.asarray([1.0, 0.0], dtype=np.float32),
                {"embedding_model": "m1", "embedding_dim": 2},
            )
            person_id = store.create_profile(emb, {"embedding_model": "m1", "embedding_dim": 2})
            query = np.asarray([0.78, np.sqrt(1 - 0.78**2)], dtype=np.float32)
            match = store.find_best_match(
                query,
                "m1",
                threshold_auto=0.84,
                threshold_uncertain=0.74,
            )

            self.assertEqual(match.match_status, "uncertain")
            self.assertEqual(match.person_id, person_id)
            self.assertGreaterEqual(match.match_similarity, 0.74)
            self.assertLess(match.match_similarity, 0.84)

    def test_no_matching_across_different_embedding_models(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = FileSpeakerIdentityStore(Path(raw_root))
            emb = store.save_embedding(
                "sess_1",
                "spk_1",
                np.asarray([1.0, 0.0], dtype=np.float32),
                {"embedding_model": "model_a", "embedding_dim": 2},
            )
            store.create_profile(emb, {"embedding_model": "model_a", "embedding_dim": 2})
            match = store.find_best_match(
                np.asarray([1.0, 0.0], dtype=np.float32),
                "model_b",
                threshold_auto=0.84,
                threshold_uncertain=0.74,
            )

            self.assertEqual(match.match_status, "new")
            self.assertIsNone(match.person_id)


if __name__ == "__main__":
    unittest.main()
