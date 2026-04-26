from __future__ import annotations

from typing import List, Tuple

LOOPBACK_SOURCE_TYPE = "System audio (WASAPI loopback)"
MIC_SOURCE_TYPE = "Microphone (input device)"


def list_loopback_devices() -> List[Tuple[str, object]]:
    out: List[Tuple[str, object]] = []
    try:
        import soundcard as sc

        mics = sc.all_microphones(include_loopback=True)
    except Exception:
        mics = []

    for mic in mics:
        if not bool(getattr(mic, "isloopback", False)):
            continue
        name = str(getattr(mic, "name", "")).strip()
        if name:
            token = getattr(mic, "id", None)
            out.append((name, token if token is not None else name))

    if not out:
        for mic in mics:
            name = str(getattr(mic, "name", "")).strip()
            if not name:
                continue
            token = getattr(mic, "id", None)
            out.append((name, token if token is not None else name))

    try:
        speaker = sc.default_speaker()
        if speaker is not None:
            default_token = speaker.name.lower()
            out.sort(key=lambda x: 0 if default_token in x[0].lower() else 1)
    except Exception:
        pass

    seen = set()
    uniq: List[Tuple[str, object]] = []
    for label, token in out:
        key = f"{str(token)}::{label.lower()}"
        if key in seen:
            continue
        seen.add(key)
        uniq.append((label, token))
    return uniq


def list_input_devices() -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    try:
        import sounddevice as sd

        devices = sd.query_devices()
    except Exception:
        devices = []

    for idx, device in enumerate(devices):
        try:
            if int(device.get("max_input_channels", 0)) <= 0:
                continue
            name = str(device.get("name", f"device-{idx}"))
            sample_rate = device.get("default_samplerate", None)
            channels = device.get("max_input_channels", None)
            label = f"[{idx}] {name} (in={channels}, sr={sample_rate})"
            out.append((label, idx))
        except Exception:
            continue
    return out
