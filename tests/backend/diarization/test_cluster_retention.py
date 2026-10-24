from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from diarization.infrastructure.diarizer import OnlineDiarizer


class ClusterRetentionTests(unittest.TestCase):
    def test_expiry_does_not_reuse_an_active_speaker_label(self) -> None:
        diarizer = OnlineDiarizer(min_segment_s=0, prune_after_s=2)
        diarizer._backend = Mock(
            embed=Mock(
                side_effect=[
                    np.array([1, 0, 0]),
                    np.array([0, 1, 0]),
                    np.array([0, 0, 1]),
                ]
            )
        )
        audio = np.ones(16000, dtype=np.float32)
        self.assertEqual(diarizer.assign(audio, ts=1)[0], 'S1')
        self.assertEqual(diarizer.assign(audio, ts=2)[0], 'S2')
        self.assertEqual(diarizer.assign(audio, ts=4)[0], 'S3')
        self.assertEqual(
            [cluster['label'] for cluster in diarizer.clusters_snapshot()],
            ['S2', 'S3'],
        )

    def test_event_history_expires_with_cluster_expiry_disabled(self) -> None:
        diarizer = OnlineDiarizer(min_segment_s=0, prune_after_s=0, window_s=2)
        diarizer._backend = Mock(embed=Mock(return_value=np.array([1, 0])))
        audio = np.ones(16000, dtype=np.float32)
        for timestamp in range(10):
            diarizer.assign(audio, ts=timestamp)
        self.assertEqual(diarizer._events, [(7, 'S1'), (8, 'S1'), (9, 'S1')])
        diarizer.window_s = 0
        diarizer.assign(audio, ts=10)
        self.assertEqual(diarizer._events, [])


if __name__ == '__main__':
    unittest.main()
