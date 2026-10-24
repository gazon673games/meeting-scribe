from __future__ import annotations

import logging
from collections.abc import Mapping

LOOPBACK_SOURCE_TYPE = 'System audio (WASAPI loopback)'
MIC_SOURCE_TYPE = 'Microphone (input device)'


_logger = logging.getLogger(__name__)


def _collect_loopback_candidates(mics: list) -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for mic in mics:
        if not bool(getattr(mic, 'isloopback', False)):
            continue
        name = str(getattr(mic, 'name', '')).strip()
        if name:
            token = getattr(mic, 'id', None)
            out.append((name, token if token is not None else name))
    if not out:
        for mic in mics:
            name = str(getattr(mic, 'name', '')).strip()
            if not name:
                continue
            token = getattr(mic, 'id', None)
            out.append((name, token if token is not None else name))
    return out


def list_loopback_devices() -> list[tuple[str, object]]:
    try:
        import soundcard as sc
    except ImportError:
        _logger.debug('SoundCard is unavailable', exc_info=True)
        return []
    try:
        mics = sc.all_microphones(include_loopback=True)
    except Exception:
        # Native discovery is an SDK boundary; do not lose its diagnostics.
        _logger.warning('Cannot enumerate loopback devices', exc_info=True)
        mics = []

    out = _collect_loopback_candidates(mics)

    try:
        speaker = sc.default_speaker()
        if speaker is not None:
            default_token = speaker.name.lower()
            out.sort(key=lambda x: 0 if default_token in x[0].lower() else 1)
    except Exception:
        _logger.debug(
            'Best-effort operation failed in list_loopback_devices',
            exc_info=True,
        )

    seen: set[tuple[str, str]] = set()
    uniq: list[tuple[str, object]] = []
    for label, token in out:
        key = (str(token), label.lower())
        if key in seen:
            continue
        seen.add(key)
        uniq.append((label, token))
    return uniq


def list_input_devices() -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception:
        _logger.warning('Cannot enumerate input devices', exc_info=True)
        devices = []

    for idx, device in enumerate(devices):
        if not isinstance(device, Mapping):
            _logger.debug('Ignoring non-mapping input device record')
            continue
        try:
            if int(device.get('max_input_channels', 0)) <= 0:
                continue
            name = str(device.get('name', f'device-{idx}'))
            sample_rate = device.get('default_samplerate', None)
            channels = device.get('max_input_channels', None)
            label = f'[{idx}] {name} (in={channels}, sr={sample_rate})'
            out.append((label, idx))
        except (TypeError, ValueError, OverflowError):
            _logger.debug('Ignoring invalid input device record', exc_info=True)
            continue
    return out
