from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tools.batch_scribe_parts.audio import read_wav_float32, to_16k_mono_wav


class Scribe:
    """Wraps ASR plus optional diarization for local batch use."""

    def __init__(
        self,
        profile,
        *,
        diar: bool = False,
        diar_backend: str = "online",
        diar_sim_threshold: float = 0.74,
        diar_device: Optional[str] = None,
        sherpa_model_path: str = "",
        sherpa_provider: str = "cpu",
        sherpa_num_threads: int = 1,
    ) -> None:
        self._profile = profile
        self._diar = diar
        self._diar_backend = diar_backend
        self._diar_sim_threshold = diar_sim_threshold
        self._diar_device = diar_device or profile.device
        self._sherpa_model_path = sherpa_model_path
        self._sherpa_provider = sherpa_provider
        self._sherpa_num_threads = sherpa_num_threads
        self._model = None

    def __enter__(self) -> "Scribe":
        from faster_whisper import WhisperModel  # type: ignore

        p = self._profile
        model_kwargs: Dict[str, Any] = {"device": p.device, "compute_type": p.compute_type}
        cpu_threads = getattr(p, "cpu_threads", None)
        num_workers = getattr(p, "num_workers", None)
        if cpu_threads is not None:
            model_kwargs["cpu_threads"] = max(0, int(cpu_threads))
        if num_workers is not None:
            model_kwargs["num_workers"] = max(1, int(num_workers))
        try:
            self._model = WhisperModel(p.model_name, **model_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Unable to load ASR model "
                f"'{p.model_name}'. The model may be missing from the local cache, "
                "the local path may be invalid, or the machine may be offline. "
                "Download the model in the app first, run once with network access, "
                "or pass a valid local model path / --models-dir."
            ) from exc
        return self

    def __exit__(self, *_) -> None:
        if self._model is not None:
            try:
                self._model.model.unload_model()
            except Exception:
                pass
            del self._model
            self._model = None

    @staticmethod
    def _segment_record(segment) -> Optional[dict]:  # noqa: ANN001
        text = (getattr(segment, "text", "") or "").strip()
        if not text:
            return None
        return {"t0": float(segment.start), "t1": float(segment.end), "text": text}

    @staticmethod
    def _word_records(segment) -> List[dict]:  # noqa: ANN001
        records: List[dict] = []
        for word in (getattr(segment, "words", None) or []):
            text = (getattr(word, "word", "") or "").strip()
            start = getattr(word, "start", None)
            end = getattr(word, "end", None)
            if text and start is not None and end is not None:
                records.append({"t0": float(start), "t1": float(end), "text": text, "unit": "word"})
        return records

    def _transcribe_kwargs(self, *, word_by_word: bool = False) -> dict:
        p = self._profile
        kwargs: dict = dict(getattr(p, "extra_transcribe_options", {}) or {})
        kwargs.update(
            beam_size=p.beam_size,
            vad_filter=p.vad_filter,
            condition_on_previous_text=p.condition_on_previous_text,
        )
        temperature = getattr(p, "temperature", None)
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if p.language:
            kwargs["language"] = p.language
        if p.initial_prompt:
            kwargs["initial_prompt"] = p.initial_prompt
        if word_by_word:
            kwargs["word_timestamps"] = True
            kwargs["without_timestamps"] = False
        return kwargs

    def _transcribe(self, wav_path: Path, *, word_by_word: bool = False) -> List[dict]:
        if self._model is None:
            raise RuntimeError("Call process() inside a 'with Scribe(...) as scribe:' block.")
        segments, _ = self._model.transcribe(str(wav_path), **self._transcribe_kwargs(word_by_word=word_by_word))
        records: List[dict] = []
        for segment in segments:
            if word_by_word:
                words = self._word_records(segment)
                if words:
                    records.extend(words)
                    continue
            record = self._segment_record(segment)
            if record is not None:
                records.append(record)
        return records

    def _transcribe_stream(self, wav_path: Path, *, word_by_word: bool = False):
        if self._model is None:
            raise RuntimeError("Call process_stream() inside a 'with Scribe(...) as scribe:' block.")
        segments, info = self._model.transcribe(str(wav_path), **self._transcribe_kwargs(word_by_word=word_by_word))
        total = getattr(info, "duration", None)
        for segment in segments:
            records = self._word_records(segment) if word_by_word else []
            if not records:
                record = self._segment_record(segment)
                records = [record] if record is not None else []
            for record in records:
                yield record, total

    def _make_online_diarizer(self):
        from diarization.infrastructure.diarizer import OnlineDiarizer  # type: ignore

        backend = {"nemo": "nemo", "sherpa_onnx": "sherpa_onnx"}.get(self._diar_backend, "resemblyzer")
        return OnlineDiarizer(
            similarity_threshold=self._diar_sim_threshold,
            backend=backend,
            device=self._diar_device,
            sherpa_model_path=self._sherpa_model_path,
            sherpa_provider=self._sherpa_provider,
            sherpa_num_threads=self._sherpa_num_threads,
        )

    def _diarize(self, audio_16k: np.ndarray, asr_segs: List[dict]) -> List[dict]:
        if self._diar_backend == "pyannote":
            return self._diarize_pyannote(audio_16k, asr_segs)
        diarizer = self._make_online_diarizer()
        sr = 16000
        result: List[dict] = []
        for seg in asr_segs:
            chunk = audio_16k[int(seg["t0"] * sr) : int(seg["t1"] * sr)]
            label, _ = diarizer.assign(chunk, ts=seg["t0"])
            result.append({**seg, "speaker": label})
        return result

    def _diarize_pyannote(self, audio_16k: np.ndarray, asr_segs: List[dict]) -> List[dict]:
        from diarization.domain.segments import pick_speaker  # type: ignore
        from diarization.infrastructure.diar_backend_pyannote import PyannoteDiarizer  # type: ignore

        timeline = PyannoteDiarizer(device=self._diar_device).diarize(audio_16k)
        return [{**seg, "speaker": pick_speaker(timeline, seg["t0"], seg["t1"])} for seg in asr_segs]

    def process(self, input_path: Path, tmp_dir: Optional[Path] = None, *, word_by_word: bool = False) -> List[dict]:
        managed = tmp_dir is None
        temp_dir = tempfile.TemporaryDirectory() if managed else None
        try:
            tmp_path = Path(temp_dir.name) if temp_dir is not None else Path(tmp_dir)
            wav = to_16k_mono_wav(input_path, tmp_path)
            segs = self._transcribe(wav, word_by_word=word_by_word)
            if self._diar and segs:
                segs = self._diarize(read_wav_float32(wav), segs)
            return segs
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def process_stream(self, input_path: Path, tmp_dir: Optional[Path] = None, *, word_by_word: bool = False):
        managed = tmp_dir is None
        temp_dir = tempfile.TemporaryDirectory() if managed else None
        try:
            tmp_path = Path(temp_dir.name) if temp_dir is not None else Path(tmp_dir)
            wav = to_16k_mono_wav(input_path, tmp_path)
            audio = read_wav_float32(wav) if self._diar else None
            pyannote_timeline = self._pyannote_timeline(audio) if self._diar and self._diar_backend == "pyannote" else None
            online_diarizer = self._make_online_diarizer() if self._diar and self._diar_backend != "pyannote" else None

            sr = 16000
            for seq, (seg, total_s) in enumerate(self._transcribe_stream(wav, word_by_word=word_by_word), 1):
                seg["_seq"] = seq
                seg["_total_s"] = total_s
                if pyannote_timeline is not None:
                    from diarization.domain.segments import pick_speaker  # type: ignore

                    seg["speaker"] = pick_speaker(pyannote_timeline, seg["t0"], seg["t1"])
                elif online_diarizer is not None and audio is not None:
                    chunk = audio[int(seg["t0"] * sr) : int(seg["t1"] * sr)]
                    label, _ = online_diarizer.assign(chunk, ts=seg["t0"])
                    seg["speaker"] = label
                yield seg
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def _pyannote_timeline(self, audio: Optional[np.ndarray]):
        from diarization.infrastructure.diar_backend_pyannote import PyannoteDiarizer  # type: ignore

        return PyannoteDiarizer(device=self._diar_device).diarize(audio)
