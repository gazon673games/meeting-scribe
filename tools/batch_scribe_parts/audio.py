from __future__ import annotations

import hashlib
import logging
import subprocess
import uuid
import wave
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)


def to_16k_mono_wav(src: Path, tmp_dir: Path) -> Path:
    metadata = src.stat()
    identity = f'{src.resolve()}:{metadata.st_size}:{metadata.st_mtime_ns}'
    digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]
    dst = tmp_dir / f'{src.stem}-{digest}__16k.wav'
    if dst.exists():
        return dst
    staged = tmp_dir / f'{uuid.uuid4().hex}.wav'
    cmd = [
        'ffmpeg',
        '-y',
        '-i',
        str(src),
        '-ac',
        '1',
        '-ar',
        '16000',
        '-sample_fmt',
        's16',
        '-f',
        'wav',
        str(staged),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            reason = result.stderr.decode(errors='replace')
            raise RuntimeError(f'ffmpeg failed for {src.name}:\n{reason}')
        staged.replace(dst)
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            _logger.warning(
                'Cannot remove audio staging file %s', staged, exc_info=True
            )
    return dst


def read_wav_float32(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), 'rb') as wf:
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    return audio
