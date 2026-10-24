from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from audio.application.mixer import render_source_block
from audio.application.source_state import SourceState
from audio.domain.formats import AudioFormat
from audio.domain.ports import AudioSource


class FilterRecoveryTests(unittest.TestCase):
    def test_repeated_filter_failure_mutes_source_without_flooding_diagnostics(
        self,
    ):
        audio_format = AudioFormat(16000, 1, blocksize=16)
        source = Mock(spec=AudioSource)
        source.name = 'mic'
        source.get_format.return_value = audio_format
        source.get_filters.return_value = []
        state = SourceState(source)
        with (
            patch(
                'audio.application.mixer.apply_filters',
                side_effect=ValueError('filter'),
            ),
            patch('audio.application.mixer._logger') as logger,
        ):
            for now in (100.0, 100.1, 101.0, 105.1):
                result = render_source_block(
                    state=state,
                    engine_format=audio_format,
                    ts_mono=now,
                    active_rms_eps=0.01,
                )
                np.testing.assert_array_equal(result.block, np.zeros((16, 1)))
        self.assertEqual(logger.exception.call_count, 2)
