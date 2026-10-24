from __future__ import annotations

import unittest
from unittest.mock import Mock

from audio.application.source_registry import SourceRegistry
from audio.domain.formats import AudioFormat
from audio.domain.ports import AudioSource


class SourceRegistryTests(unittest.TestCase):
    def test_format_failure_leaves_no_partially_registered_source(self) -> None:
        registry = SourceRegistry(AudioFormat(sample_rate=48000, channels=2))
        source = Mock(spec=AudioSource)
        source.name = 'mic'
        source.get_format.side_effect = RuntimeError('format unavailable')

        with self.assertRaisesRegex(RuntimeError, 'format unavailable'):
            registry.add_source(source, running=False)

        self.assertFalse(registry.has_sources())
        self.assertEqual(registry.source_items(), [])
        source.get_format.side_effect = None
        source.get_format.return_value = AudioFormat(
            sample_rate=48000, channels=2
        )
        registry.add_source(source, running=False)
        self.assertEqual(registry.sources, [source])

    def test_collection_snapshots_cannot_remove_owned_sources(self) -> None:
        format = AudioFormat(sample_rate=48000, channels=2)
        registry = SourceRegistry(format)
        source = Mock(spec=AudioSource)
        source.name = 'mic'
        source.get_format.return_value = format
        registry.add_source(source, running=False)

        registry.sources.clear()
        registry.state.clear()

        self.assertEqual(registry.sources, [source])
        self.assertIs(registry.get_state('mic').src, source)


if __name__ == '__main__':
    unittest.main()
