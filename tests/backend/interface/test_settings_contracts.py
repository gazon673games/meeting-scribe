from __future__ import annotations

import unittest

from interface.session_controller_parts.artifacts import build_session_speakers
from interface.session_controller_parts.settings import asr_settings_from_params


class SettingsContractTests(unittest.TestCase):
    def test_rejects_inverted_overload_thresholds(self):
        with self.assertRaisesRegex(ValueError, 'Overload thresholds'):
            asr_settings_from_params(
                {'overloadExitQsize': 20, 'overloadEnterQsize': 10}
            )

    def test_rejects_uncertain_threshold_above_automatic_matching(self):
        with self.assertRaisesRegex(ValueError, 'speaker threshold'):
            asr_settings_from_params(
                {
                    'speakerIdentity': {
                        'auto_match_threshold': 0.7,
                        'uncertain_match_threshold': 0.9,
                    }
                }
            )

    def test_composite_speaker_key_does_not_collide_on_delimiter_text(self):
        lines = [
            {'stream': 'a\x1fb', 'speaker': 'c'},
            {'stream': 'a', 'speaker': 'b\x1fc'},
        ]
        speakers = build_session_speakers(lines)
        self.assertEqual(len(speakers), 2)
        self.assertNotEqual(
            lines[0]['session_speaker_id'], lines[1]['session_speaker_id']
        )
