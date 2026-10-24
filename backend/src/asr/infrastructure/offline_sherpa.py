from __future__ import annotations

import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

import soundfile as sf

from application.native_asr_models import native_asr_model_dir
from asr.infrastructure.audio_utils import resample_linear, stereo_to_mono
from asr.infrastructure.worker_sherpa_offline import SherpaOfflineASR

if TYPE_CHECKING:
    from asr.infrastructure.offline_runner import OfflineProfile


def transcribe_offline_onnx(
    *,
    project_root: Path,
    wav_path: Path,
    out_txt: Path,
    out_jsonl: Path | None,
    profile: OfflineProfile,
) -> Path:
    parts: list[str] = []
    with ExitStack() as stack:
        backend = SherpaOfflineASR(
            model_dir=native_asr_model_dir(
                profile.model_name,
                models_dir=profile.models_dir,
                project_root=project_root,
            ),
            model_name=profile.model_name,
            language=profile.language,
            device=profile.device,
            compute_type=profile.compute_type,
            cpu_threads=profile.cpu_threads or 0,
            num_workers=profile.num_workers or 1,
            beam_size=profile.beam_size,
            initial_prompt=profile.initial_prompt,
            hotwords=None,
        )
        stack.callback(backend.close)
        source = stack.enter_context(sf.SoundFile(wav_path))
        records = (
            stack.enter_context(out_jsonl.open('a', encoding='utf-8'))
            if out_jsonl is not None
            else None
        )
        offset_s = 0.0
        for block in source.blocks(
            blocksize=source.samplerate * 30, dtype='float32', always_2d=True
        ):
            duration_s = len(block) / source.samplerate
            samples = resample_linear(
                stereo_to_mono(block), source.samplerate, 16000
            )
            text = backend.transcribe(samples)['text'].strip()
            if text:
                parts.append(text)
                if records is not None:
                    records.write(
                        json.dumps(
                            {
                                'type': 'offline_segment',
                                't0': offset_s,
                                't1': offset_s + duration_s,
                                'text': text,
                                'ts': time.time(),
                            },
                            ensure_ascii=False,
                        )
                        + '\n'
                    )
            offset_s += duration_s
    text = '\n'.join(parts)
    out_txt.write_text(text + ('\n' if text else ''), encoding='utf-8')
    return out_txt
