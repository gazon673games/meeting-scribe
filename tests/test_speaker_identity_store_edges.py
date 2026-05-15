from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from identity.infrastructure.file_store import FileSpeakerIdentityStore


class SpeakerIdentityStoreEdgeCaseTests(unittest.TestCase):
    def test_identity_store_skips_invalid_records_and_updates_profile_centroids(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            store = FileSpeakerIdentityStore(Path(raw_root))

            with self.assertRaises(FileNotFoundError):
                store.load_embedding("missing")
            with self.assertRaises(ValueError):
                store._embedding_path("")
            self.assertEqual(store.find_best_match(np.array([1.0, 0.0]), "", 0.8, 0.7).match_status, "new")

            store._profiles_path.write_text(
                "\nnot-json\n[]\n"
                '{"person_id":"","embedding_ids":["x"]}\n'
                '{"person_id":"p1","embedding_ids":["missing"],"centroid_embedding_id":"missing","embedding_model":"m","embedding_dim":2}\n',
                encoding="utf-8",
            )
            profiles = store.list_profiles()
            self.assertEqual(len(profiles), 1)
            self.assertEqual(profiles[0].person_id, "p1")
            self.assertEqual(store.find_best_match(np.array([1.0, 0.0]), "m", 0.8, 0.7).match_status, "new")

            emb1 = store.save_embedding("session", "S1", np.array([1.0, 0.0]), {"embedding_model": "m"})
            person_id = store.create_profile(emb1, {"embedding_model": "m", "embedding_dim": 2})
            emb2 = store.save_embedding("session", "S2", np.array([0.0, 1.0]), {"embedding_model": "m"})

            store.link_session_speaker("session", "S2", "unknown", emb2, 0.5, "ignored")
            store.link_session_speaker("session", "S2", person_id, emb2, 0.5, "accepted")
            store.update_profile_centroid("unknown")
            store.update_profile_centroid(person_id)

            latest = store._latest_profile_records()[person_id]
            self.assertIn(emb2, latest["embedding_ids"])
            self.assertTrue(str(latest["centroid_embedding_id"]).startswith("centroid_"))
            self.assertEqual(store.find_best_match(np.array([1.0, 0.0]), "other", 0.8, 0.7).match_status, "new")

            zero = store.save_embedding("session", "S3", np.array([0.0, 0.0]), {"embedding_model": "m"})
            np.testing.assert_allclose(store.load_embedding(zero), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
