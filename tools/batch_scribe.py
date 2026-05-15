#!/usr/bin/env python3
"""
batch_scribe.py — Standalone batch transcription + optional speaker diarization.

Uses the same ASR profiles, model config, and diarization backends as the GUI,
but without any GUI — just a script you run from anywhere.

Accepts any format ffmpeg can read (WAV, MP3, MP4, MKV, …).
Outputs SRT, TXT, or JSONL.

CLI:
    python tools/batch_scribe.py recordings/*.wav --profile Quality --language ru
    python tools/batch_scribe.py meeting.mp4 --diar --out-dir ./out
    python tools/batch_scribe.py call.wav --profile Custom --beam-size 8 --compute-type float16

Library:
    import sys; sys.path.insert(0, "/path/to/project/tools")
    from batch_scribe import build_profile, Scribe, write_output
    from pathlib import Path

    profile = build_profile("Quality", model="large-v3", language="ru")
    with Scribe(profile, diar=True) as scribe:
        for f in Path("audio").glob("*.wav"):
            segs = scribe.process(f)
            write_output(segs, f.with_suffix(".srt"), "srt")
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Bootstrap: add backend/src so project modules are importable
# ---------------------------------------------------------------------------

def _bootstrap(project_root: Optional[str] = None) -> None:
    if project_root:
        src = Path(project_root) / "backend" / "src"
    else:
        src = Path(__file__).resolve().parent.parent / "backend" / "src"
    src = src.resolve()
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_bootstrap()


# ---------------------------------------------------------------------------
# Profile helpers — same profiles as in the GUI
# ---------------------------------------------------------------------------

PROFILES = ("Quality", "Balanced", "Realtime", "Ultra Fast", "Custom")


def build_profile(
    profile_name: str = "Quality",
    *,
    model: str = "large-v3",
    device: str = "cuda",
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    vad_filter: bool = True,
    condition_on_previous_text: bool = True,
    # Custom overrides — only applied when profile_name is "Custom"
    # or to explicitly override profile defaults:
    compute_type: Optional[str] = None,
    beam_size: Optional[int] = None,
):
    """
    Build an OfflineProfile using the same profile presets as the GUI.

    Quality   → float16, beam_size=6  (best quality, slower)
    Balanced  → float16, beam_size=5
    Realtime  → int8_float16, beam_size=1  (fastest, lower quality)
    Custom    → you must pass compute_type and beam_size explicitly
    """
    from asr.infrastructure.offline_runner import OfflineProfile  # type: ignore
    from application.asr_profiles import profile_defaults  # type: ignore

    defaults = profile_defaults(profile_name)

    ct = compute_type or defaults.get("compute_type", "float16")
    bs = beam_size if beam_size is not None else defaults.get("beam_size", 6)

    return OfflineProfile(
        model_name=model,
        device=device,
        compute_type=ct,
        beam_size=bs,
        language=language,
        initial_prompt=initial_prompt,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
    )


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def _to_16k_mono_wav(src: Path, tmp_dir: Path) -> Path:
    dst = tmp_dir / (src.stem + "__16k.wav")
    if dst.exists():
        return dst
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ac", "1", "-ar", "16000",
        "-sample_fmt", "s16",
        "-f", "wav", str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {src.name}:\n"
            f"{result.stderr.decode(errors='replace')}"
        )
    return dst


def _read_wav_float32(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        n_ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    return audio


# ---------------------------------------------------------------------------
# Scribe — loads model once, processes many files
# ---------------------------------------------------------------------------

class Scribe:
    """
    Wraps the project's ASR + diarization infrastructure for batch use.

    Model is loaded once in __enter__ and released in __exit__.
    Diarization backends are the same as what the GUI uses.

    diar_backend options:
      "online"      — local resemblyzer embeddings (default, no extra install)
      "nemo"        — local NeMo TitaNet embeddings
      "sherpa_onnx" — local sherpa-onnx embeddings
      "pyannote"    — local pyannote.audio (best quality, needs HF token on first use)

    All backends are fully local — no internet required after model download.
    "online" here means "streaming/incremental", not "internet-based".
    """

    def __init__(
        self,
        profile,                            # OfflineProfile from build_profile()
        *,
        diar: bool = False,
        diar_backend: str = "online",       # same names as GUI
        diar_sim_threshold: float = 0.74,
        diar_device: Optional[str] = None,  # None → same as profile.device
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

    # --- context manager ---

    def __enter__(self) -> "Scribe":
        from faster_whisper import WhisperModel  # type: ignore
        p = self._profile
        self._model = WhisperModel(p.model_name, device=p.device, compute_type=p.compute_type)
        return self

    def __exit__(self, *_) -> None:
        if self._model is not None:
            try:
                self._model.model.unload_model()
            except Exception:
                pass
            del self._model
            self._model = None

    # --- transcription ---

    def _transcribe_kwargs(self) -> dict:
        p = self._profile
        kwargs: dict = dict(
            beam_size=p.beam_size,
            vad_filter=p.vad_filter,
            condition_on_previous_text=p.condition_on_previous_text,
        )
        if p.language:
            kwargs["language"] = p.language
        if p.initial_prompt:
            kwargs["initial_prompt"] = p.initial_prompt
        return kwargs

    def _transcribe(self, wav_path: Path) -> List[dict]:
        if self._model is None:
            raise RuntimeError("Call process() inside a 'with Scribe(...) as scribe:' block.")
        segments_gen, _ = self._model.transcribe(str(wav_path), **self._transcribe_kwargs())
        return [
            {"t0": float(s.start), "t1": float(s.end), "text": s.text.strip()}
            for s in segments_gen if s.text.strip()
        ]

    def _transcribe_stream(self, wav_path: Path):
        """Generator: yields (seg_dict, total_duration_or_None) one segment at a time."""
        if self._model is None:
            raise RuntimeError("Call process_stream() inside a 'with Scribe(...) as scribe:' block.")
        segments_gen, info = self._model.transcribe(str(wav_path), **self._transcribe_kwargs())
        total = getattr(info, "duration", None)
        for s in segments_gen:
            text = (s.text or "").strip()
            if text:
                yield {"t0": float(s.start), "t1": float(s.end), "text": text}, total

    # --- diarization ---

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
            chunk = audio_16k[int(seg["t0"] * sr): int(seg["t1"] * sr)]
            label, _ = diarizer.assign(chunk, ts=seg["t0"])
            result.append({**seg, "speaker": label})
        return result

    def _diarize_pyannote(self, audio_16k: np.ndarray, asr_segs: List[dict]) -> List[dict]:
        from diarization.infrastructure.diar_backend_pyannote import PyannoteDiarizer  # type: ignore
        from diarization.domain.segments import pick_speaker  # type: ignore
        timeline = PyannoteDiarizer(device=self._diar_device).diarize(audio_16k)
        return [{**seg, "speaker": pick_speaker(timeline, seg["t0"], seg["t1"])} for seg in asr_segs]

    # --- public API ---

    def process(self, input_path: Path, tmp_dir: Optional[Path] = None) -> List[dict]:
        """
        Transcribe (and optionally diarize) one file.
        Returns full list of {t0, t1, text[, speaker]} dicts.
        """
        managed = tmp_dir is None
        _tmp = None
        if managed:
            _tmp = tempfile.TemporaryDirectory()
            tmp_dir = Path(_tmp.name)
        try:
            wav = _to_16k_mono_wav(input_path, tmp_dir)
            segs = self._transcribe(wav)
            if self._diar and segs:
                segs = self._diarize(_read_wav_float32(wav), segs)
            return segs
        finally:
            if managed and _tmp is not None:
                _tmp.cleanup()

    def process_stream(self, input_path: Path, tmp_dir: Optional[Path] = None):
        """
        Generator: yields one segment at a time as transcription progresses.

        Each yielded dict: {t0, t1, text[, speaker], _seq, _total_s}
          _seq     — 1-based segment index
          _total_s — total audio duration in seconds (float or None)

        Diarization (if enabled) is applied per-segment on the fly.
        For pyannote backend, the full audio is diarized upfront before streaming starts.

        Usage:
            with Scribe(profile, diar=True, ...) as scribe:
                for seg in scribe.process_stream(Path("file.mp4")):
                    print(seg)
        """
        managed = tmp_dir is None
        _tmp = None
        if managed:
            _tmp = tempfile.TemporaryDirectory()
            tmp_dir = Path(_tmp.name)
        try:
            wav = _to_16k_mono_wav(input_path, tmp_dir)
            audio = _read_wav_float32(wav) if self._diar else None

            # pyannote needs full audio upfront — pre-run before streaming ASR
            pyannote_timeline = None
            if self._diar and self._diar_backend == "pyannote":
                from diarization.infrastructure.diar_backend_pyannote import PyannoteDiarizer  # type: ignore
                pyannote_timeline = PyannoteDiarizer(device=self._diar_device).diarize(audio)

            # online/nemo/sherpa: keep one diarizer instance alive across all segments
            online_diarizer = None
            if self._diar and self._diar_backend != "pyannote":
                online_diarizer = self._make_online_diarizer()

            sr = 16000
            seq = 0
            for seg, total_s in self._transcribe_stream(wav):
                seq += 1
                seg["_seq"] = seq
                seg["_total_s"] = total_s

                if self._diar:
                    if pyannote_timeline is not None:
                        from diarization.domain.segments import pick_speaker  # type: ignore
                        seg["speaker"] = pick_speaker(pyannote_timeline, seg["t0"], seg["t1"])
                    elif online_diarizer is not None and audio is not None:
                        chunk = audio[int(seg["t0"] * sr): int(seg["t1"] * sr)]
                        label, _ = online_diarizer.assign(chunk, ts=seg["t0"])
                        seg["speaker"] = label

                yield seg
        finally:
            if managed and _tmp is not None:
                _tmp.cleanup()


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _srt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_output(segs: List[dict], out_path: Path, fmt: str) -> None:
    """Write segments to SRT, TXT, or JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "txt":
        lines = []
        for seg in segs:
            spk = seg.get("speaker", "")
            lines.append((f"[{spk}] " if spk else "") + seg["text"])
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    elif fmt == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for seg in segs:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")

    else:  # srt
        blocks: List[str] = []
        for i, seg in enumerate(segs, 1):
            spk = seg.get("speaker", "")
            prefix = f"[{spk}] " if spk else ""
            blocks.append(
                f"{i}\n"
                f"{_srt_ts(seg['t0'])} --> {_srt_ts(seg['t1'])}\n"
                f"{prefix}{seg['text']}"
            )
        out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def append_srt_segment(seg: dict, index: int, f) -> None:
    """
    Write one SRT block to an already-open file (for incremental streaming writes).

    index — 1-based segment number
    f     — open file object (text mode, utf-8)

    Adds a blank line between blocks automatically.
    """
    spk = seg.get("speaker", "")
    prefix = f"[{spk}] " if spk else ""
    block = (
        f"{index}\n"
        f"{_srt_ts(seg['t0'])} --> {_srt_ts(seg['t1'])}\n"
        f"{prefix}{seg['text']}\n"
    )
    if index > 1:
        f.write("\n")
    f.write(block)
    f.flush()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="batch_scribe",
        description="Batch transcription using the same ASR + diarization as the GUI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
profiles (same as GUI):
  Quality    — float16, beam_size=6   (best quality, recommended for batch)
  Balanced   — float16, beam_size=5
  Realtime   — int8_float16, beam_size=1  (fastest)
  Ultra Fast — int8_float16, beam_size=1  (streaming presets, beam stays 1)
  Custom     — set --compute-type and --beam-size explicitly

examples:
  python tools/batch_scribe.py *.wav --profile Quality --language ru
  python tools/batch_scribe.py meeting.mp4 --diar --out-dir ./out
  python tools/batch_scribe.py call.wav --profile Custom --beam-size 8 --compute-type float16
  python tools/batch_scribe.py interview.mp4 --diar --diar-backend pyannote --format txt
        """,
    )

    p.add_argument("files", nargs="+", type=Path, metavar="FILE")

    asr = p.add_argument_group("ASR")
    asr.add_argument("--profile", default="Quality", choices=list(PROFILES),
                     help="ASR quality preset (default: Quality)")
    asr.add_argument("--model", default="large-v3",
                     help="Whisper model: tiny/base/small/medium/large-v2/large-v3 (default: large-v3)")
    asr.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    asr.add_argument("--language", default=None, metavar="LANG",
                     help="Language code: ru, en, de, … (default: auto-detect)")
    asr.add_argument("--initial-prompt", default=None, metavar="TEXT",
                     help="Domain hint to improve terminology")
    asr.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    # Custom profile overrides
    asr.add_argument("--compute-type", default=None,
                     choices=["float16", "int8_float16", "int8", "float32"],
                     help="Override compute type (used with --profile Custom)")
    asr.add_argument("--beam-size", type=int, default=None,
                     help="Override beam size (used with --profile Custom)")

    diar = p.add_argument_group("Diarization (same backends as GUI, all local)")
    diar.add_argument("--diar", action="store_true", help="Enable speaker diarization")
    diar.add_argument("--diar-backend", default="online",
                      choices=["online", "pyannote", "nemo", "sherpa_onnx"],
                      help=(
                          "online      — local resemblyzer embeddings (default)\n"
                          "nemo        — local NeMo TitaNet embeddings\n"
                          "sherpa_onnx — local sherpa-onnx embeddings\n"
                          "pyannote    — local pyannote.audio (best quality)"
                      ))
    diar.add_argument("--diar-threshold", type=float, default=0.74, metavar="T",
                      help="Cosine similarity threshold for online/nemo/sherpa (default: 0.74)")
    diar.add_argument("--diar-device", default=None,
                      help="Device for diarizer (default: same as --device)")
    diar.add_argument("--sherpa-model-path", default="", metavar="PATH")
    diar.add_argument("--sherpa-provider", default="cpu", choices=["cpu", "cuda", "coreml"])
    diar.add_argument("--sherpa-threads", type=int, default=1)

    out = p.add_argument_group("Output")
    out.add_argument("--format", default="srt", choices=["srt", "txt", "jsonl"])
    out.add_argument("--out-dir", type=Path, default=None,
                     help="Output directory (default: same dir as each input file)")

    p.add_argument("--project-root", default=None,
                   help="Project root path (auto-detected from script location by default)")

    return p


def _out_path(src: Path, fmt: str, out_dir: Optional[Path]) -> Path:
    ext = {"srt": ".srt", "txt": ".txt", "jsonl": ".jsonl"}[fmt]
    return (out_dir / (src.stem + ext)) if out_dir else src.with_suffix(ext)


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.project_root:
        _bootstrap(args.project_root)

    profile = build_profile(
        args.profile,
        model=args.model,
        device=args.device,
        language=args.language,
        initial_prompt=args.initial_prompt,
        vad_filter=not args.no_vad,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
    )

    files = [f for f in (Path(x) for x in args.files) if f.exists() or
             print(f"[SKIP] not found: {f}", file=sys.stderr) or False]
    if not files:
        print("No valid input files.", file=sys.stderr)
        sys.exit(1)

    scribe_kwargs = dict(
        diar=args.diar,
        diar_backend=args.diar_backend,
        diar_sim_threshold=args.diar_threshold,
        diar_device=args.diar_device,
        sherpa_model_path=args.sherpa_model_path,
        sherpa_provider=args.sherpa_provider,
        sherpa_num_threads=args.sherpa_threads,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with Scribe(profile, **scribe_kwargs) as scribe:
            for src in files:
                print(f"[{src.name}] transcribing...", end=" ", flush=True)
                try:
                    segs = scribe.process(src, tmp_dir)
                    out = _out_path(src, args.format, args.out_dir)
                    write_output(segs, out, args.format)
                    print(f"→ {out}  ({len(segs)} segments)")
                except Exception as exc:
                    print(f"ERROR: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
