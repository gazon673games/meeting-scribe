from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np


def to_16k_mono_wav(src: Path, tmp_dir: Path) -> Path:
    dst = tmp_dir / (src.stem + "__16k.wav")
    if dst.exists():
        return dst
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        "-f",
        "wav",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src.name}:\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    return dst


def read_wav_float32(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    return audio

