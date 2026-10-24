from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import Mock, patch

import numpy as np

from audio.domain.formats import AudioFormat
from audio.infrastructure.sources import wasapi_loopback as loopback


@dataclass(frozen=True)
class _Device:
    id: str
    name: str
    isloopback: bool = True


def _source(device: str | int | None = None) -> loopback.WasapiLoopbackSource:
    return loopback.WasapiLoopbackSource(
        'system', AudioFormat(48000, 2), device=device
    )


class WasapiLoopbackTests(unittest.TestCase):
    def test_device_lookup_prefers_id_then_exact_name_then_substring(self):
        devices = [
            _Device('first', 'Studio Speakers Extra'),
            _Device('second', 'Studio Speakers'),
            _Device('Studio Speakers', 'Other'),
        ]
        soundcard = Mock()
        self.assertIs(
            loopback._find_device(soundcard, devices, 'Studio Speakers'),
            devices[2],
        )
        self.assertIs(
            loopback._find_device(soundcard, devices, ' studio speakers '),
            devices[1],
        )
        self.assertIs(
            loopback._find_device(soundcard, devices, 'STUDIO'), devices[0]
        )
        soundcard.get_microphone.assert_not_called()

    def test_sdk_fallback_normalizes_identifier_and_preserves_failures(self):
        soundcard = Mock()
        loopback._find_device(soundcard, [], 12)
        soundcard.get_microphone.assert_called_once_with(
            '12', include_loopback=True
        )
        soundcard.get_microphone.side_effect = RuntimeError('SDK failed')
        with self.assertRaisesRegex(RuntimeError, 'SDK failed'):
            loopback._find_device(soundcard, [], 'missing')
        self.assertEqual(soundcard.get_microphone.call_count, 2)

    def test_default_device_uses_loopbacks_and_reports_no_capture_devices(self):
        microphone = _Device('mic', 'Microphone', isloopback=False)
        speaker = _Device('speaker', 'Speakers')
        soundcard = Mock()
        soundcard.default_speaker.return_value = speaker
        soundcard.all_microphones.return_value = [microphone, speaker]
        with patch.dict('sys.modules', {'soundcard': soundcard}):
            self.assertIs(_source()._pick_loopback_mic(), speaker)
            soundcard.all_microphones.return_value = []
            with self.assertRaisesRegex(RuntimeError, 'No capture devices'):
                _source()._pick_loopback_mic()

    def test_capture_converts_samples_and_closes_the_recorder(self):
        source = _source()
        source._on_audio = Mock(side_effect=lambda *_: source._stop.set())
        microphone = Mock()
        recorder = microphone.recorder.return_value
        recorder.__enter__ = Mock(return_value=recorder)
        recorder.__exit__ = Mock()
        recorder.record.side_effect = [None, [[0.25, -0.5]]]
        with patch.object(
            source, '_pick_loopback_mic', return_value=microphone
        ):
            source._run()
        name, samples = source._on_audio.call_args.args
        self.assertEqual(name, 'system')
        self.assertEqual(samples.dtype, np.float32)
        np.testing.assert_array_equal(samples, [[0.25, -0.5]])
        recorder.__exit__.assert_called_once_with(None, None, None)

    def test_failed_capture_preserves_primary_error_when_callback_fails(self):
        source = _source()
        source.set_error_callback(Mock(side_effect=ValueError('callback bug')))
        with (
            patch.object(
                source, '_pick_loopback_mic', side_effect=RuntimeError('device')
            ),
            self.assertLogs(loopback.__name__, level='ERROR') as logs,
        ):
            source._run()
        self.assertEqual(source.get_last_error(), 'RuntimeError: device')
        self.assertEqual(len(logs.records), 2)

    def test_stop_retains_live_worker_and_blocks_restart_until_it_exits(self):
        source = _source()
        worker = Mock()
        worker.is_alive.return_value = True
        source._thread = worker
        stop_signal = source._stop
        source.stop()
        self.assertIs(source._thread, worker)
        self.assertTrue(stop_signal.is_set())
        with self.assertRaisesRegex(RuntimeError, 'worker is stopping'):
            source.start(Mock())
        self.assertIs(source._stop, stop_signal)

        worker.is_alive.return_value = False
        with patch.object(loopback.threading, 'Thread') as factory:
            source.start(Mock())
        factory.return_value.start.assert_called_once()
        self.assertIsNot(source._stop, stop_signal)
        self.assertTrue(stop_signal.is_set())
        self.assertFalse(source._stop.is_set())

    def test_start_failure_releases_worker_handle(self):
        source = _source()
        with patch.object(loopback.threading, 'Thread') as factory:
            factory.return_value.start.side_effect = RuntimeError('thread')
            with self.assertRaisesRegex(RuntimeError, 'thread'):
                source.start(Mock())
        self.assertIsNone(source._thread)


if __name__ == '__main__':
    unittest.main()
