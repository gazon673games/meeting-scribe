from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from application.native_asr_models import (
    GIGAAM_MODEL_ID,
    SHERPA_OFFLINE_MODEL_IDS,
    native_asr_model_dir,
)

_logger = logging.getLogger(__name__)


@dataclass
class OfflineProfile:
    model_name: str = 'large-v3'
    device: str = 'cuda'  # "cuda" or "cpu"
    compute_type: str = 'float16'  # "float16" / "int8_float16" / etc
    beam_size: int = 6
    language: str | None = None  # None => auto
    initial_prompt: str | None = None

    vad_filter: bool = True
    condition_on_previous_text: bool = True
    batch_size: int = 4
    cpu_threads: int | None = None
    num_workers: int | None = None
    temperature: float | None = None
    extra_transcribe_options: dict[str, Any] = field(default_factory=dict)
    models_dir: Path | None = None

    # faster-whisper defaults are fine; keep knobs minimal


def _collect_transcript_segments(
    model,
    wav_path: Path,
    profile: OfflineProfile,
    out_jsonl: Path | None,
    *,
    batch_size: int | None = None,
) -> list[str]:
    import json

    transcribe_kwargs = {
        'language': profile.language,
        'beam_size': int(profile.beam_size),
        'initial_prompt': profile.initial_prompt,
        'vad_filter': bool(profile.vad_filter),
        'condition_on_previous_text': bool(profile.condition_on_previous_text),
    }
    if batch_size is not None:
        transcribe_kwargs['batch_size'] = max(1, int(batch_size))
    segments, _ = model.transcribe(str(wav_path), **transcribe_kwargs)
    txt_parts: list[str] = []
    fj = (
        out_jsonl.open('a', encoding='utf-8') if out_jsonl is not None else None
    )
    try:
        for seg in segments:
            s_text = (seg.text or '').strip()
            if not s_text:
                continue
            txt_parts.append(s_text)
            if fj is not None:
                fj.write(
                    json.dumps(
                        {
                            'type': 'offline_segment',
                            't0': float(seg.start),
                            't1': float(seg.end),
                            'text': s_text,
                            'ts': time.time(),
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )
    finally:
        if fj is not None:
            try:
                fj.close()
            except OSError:
                _logger.debug(
                    'Best-effort operation failed in _collect_transcript_segments',
                    exc_info=True,
                )
    return txt_parts


class OfflineRunner:
    """
    Offline pass over a saved WAV for higher quality transcript.
    Writes:
      - <out_txt>: plain text transcript
      - <out_jsonl> (optional): structured segments
    """

    def __init__(self, *, project_root: Path):
        self.project_root = Path(project_root)

    def run(
        self,
        wav_path: Path,
        *,
        out_txt: Path,
        out_jsonl: Path | None = None,
        profile: OfflineProfile | None = None,
    ) -> Path:
        if profile is None:
            profile = OfflineProfile()
        wav_path = Path(wav_path)
        out_txt = Path(out_txt)
        if out_jsonl is not None:
            out_jsonl = Path(out_jsonl)

        out_txt.parent.mkdir(parents=True, exist_ok=True)
        if out_jsonl is not None:
            out_jsonl.parent.mkdir(parents=True, exist_ok=True)

        if profile.model_name == GIGAAM_MODEL_ID:
            return self._run_gigaam(
                wav_path=wav_path,
                out_txt=out_txt,
                out_jsonl=out_jsonl,
                models_dir=profile.models_dir,
            )

        if profile.model_name in SHERPA_OFFLINE_MODEL_IDS:
            from asr.infrastructure.offline_sherpa import (
                transcribe_offline_onnx,
            )

            return transcribe_offline_onnx(
                project_root=self.project_root,
                wav_path=wav_path,
                out_txt=out_txt,
                out_jsonl=out_jsonl,
                profile=profile,
            )

        try:
            import faster_whisper  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                'Offline pass requires faster-whisper.\n'
                'Install:\n'
                '  pip install faster-whisper\n'
                f'Import error: {type(error).__name__}: {error}'
            ) from error

        model = faster_whisper.WhisperModel(
            profile.model_name,
            device=profile.device,
            compute_type=profile.compute_type,
        )
        try:
            batched_pipeline_type = getattr(
                faster_whisper, 'BatchedInferencePipeline', None
            )
            transcriber = (
                batched_pipeline_type(model=model)
                if batched_pipeline_type is not None
                else model
            )
            txt_parts = _collect_transcript_segments(
                transcriber,
                wav_path,
                profile,
                out_jsonl,
                batch_size=profile.batch_size
                if batched_pipeline_type is not None
                else None,
            )
        finally:
            try:
                model.model.unload_model()
            except Exception:
                _logger.debug(
                    'Best-effort operation failed in run', exc_info=True
                )
            del model

        full_text = '\n'.join(txt_parts).strip()
        out_txt.write_text(
            full_text + ('\n' if full_text else ''), encoding='utf-8'
        )
        return out_txt

    def _run_gigaam(
        self,
        *,
        wav_path: Path,
        out_txt: Path,
        out_jsonl: Path | None,
        models_dir: Path | None = None,
    ) -> Path:
        import json

        import numpy as np
        import soundfile as sf  # type: ignore
        import torch

        from asr.infrastructure.audio_utils import (
            resample_linear,
            stereo_to_mono,
        )

        try:
            from transformers import AutoModel  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                'GigaAM offline pass requires transformers, torch, torchaudio, hydra-core, and sentencepiece'
            ) from error

        model_dir = native_asr_model_dir(
            GIGAAM_MODEL_ID,
            models_dir=models_dir,
            project_root=self.project_root,
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = AutoModel.from_pretrained(
            str(model_dir), trust_remote_code=True
        )
        core = None
        records = []
        try:
            model.to(device).eval()
            audio, sample_rate = sf.read(
                str(wav_path), dtype='float32', always_2d=True
            )
            mono = stereo_to_mono(np.asarray(audio, dtype=np.float32))
            samples = resample_linear(mono, int(sample_rate), 16000)
            chunk_samples = 20 * 16000
            overlap_samples = 16000
            step_samples = chunk_samples - overlap_samples
            core = model.model
            previous_text = ''
            with torch.inference_mode():
                for start in range(0, max(1, int(samples.size)), step_samples):
                    chunk = samples[start : start + chunk_samples]
                    if chunk.size < 1600:
                        break
                    waveform = (
                        torch.from_numpy(chunk)
                        .to(core._device)
                        .to(core._dtype)
                        .unsqueeze(0)
                    )
                    length = torch.full(
                        [1], waveform.shape[-1], device=core._device
                    )
                    encoded, encoded_len = core.forward(waveform, length)
                    raw_text = str(
                        core.decoding.decode(core.head, encoded, encoded_len)[0]
                        or ''
                    ).strip()
                    text = _remove_text_overlap(previous_text, raw_text)
                    previous_text = raw_text
                    if text:
                        records.append(
                            {
                                'transcription': text,
                                'boundaries': (
                                    start / 16000.0,
                                    (start + chunk.size) / 16000.0,
                                ),
                            }
                        )
        finally:
            del core
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        texts = [str(record['transcription']) for record in records]
        out_txt.write_text(
            '\n'.join(texts) + ('\n' if texts else ''), encoding='utf-8'
        )
        if out_jsonl is not None:
            with out_jsonl.open('a', encoding='utf-8') as stream:
                for record in records:
                    boundaries = record['boundaries']
                    stream.write(
                        json.dumps(
                            {
                                'type': 'offline_segment',
                                't0': boundaries[0],
                                't1': boundaries[1],
                                'text': record['transcription'],
                                'ts': time.time(),
                            },
                            ensure_ascii=False,
                        )
                        + '\n'
                    )
        return out_txt


def _remove_text_overlap(
    previous: str, current: str, *, max_words: int = 24
) -> str:
    previous_words = str(previous or '').split()
    current_words = str(current or '').split()
    limit = min(max_words, len(previous_words), len(current_words))
    for size in range(limit, 0, -1):
        left = [
            word.casefold().strip('.,!?;:') for word in previous_words[-size:]
        ]
        right = [
            word.casefold().strip('.,!?;:') for word in current_words[:size]
        ]
        if left == right:
            return ' '.join(current_words[size:]).strip()
    return str(current or '').strip()


OfflineASRRunner = OfflineRunner
