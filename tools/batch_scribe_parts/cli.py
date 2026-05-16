from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.batch_scribe_parts.constants import OUTPUT_FORMATS, PROFILES
from tools.batch_scribe_parts.entrypoint import scribe_kwargs_from_request
from tools.batch_scribe_parts.output import write_output
from tools.batch_scribe_parts.profiles import profile_from_request
from tools.batch_scribe_parts.request import BatchScribeRequest
from tools.batch_scribe_parts.scribe import Scribe


def _coerce_cli_value(raw: str) -> Any:
    value = str(raw).strip()
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_asr_options(values: Optional[List[str]]) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"ASR option must use KEY=VALUE syntax: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"ASR option key is empty: {item}")
        options[key] = _coerce_cli_value(value)
    return options


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="batch_scribe",
        description="Batch transcription using the same ASR + diarization as the GUI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
profiles (same as GUI):
  Quality    - float16, beam_size=6   (best quality, recommended for batch)
  Balanced   - float16, beam_size=5
  Realtime   - int8_float16, beam_size=1  (fastest)
  Ultra Fast - int8_float16, beam_size=1  (streaming presets, beam stays 1)
  Custom     - set --compute-type and --beam-size explicitly

examples:
  python tools/batch_scribe.py *.wav --profile Quality --language ru
  python tools/batch_scribe.py meeting.mp4 --diar --out-dir ./out
  python tools/batch_scribe.py call.wav --profile Custom --beam-size 8 --compute-type float16
  python tools/batch_scribe.py interview.mp4 --diar --diar-backend pyannote --format txt
        """,
    )
    p.add_argument("files", nargs="+", type=Path, metavar="FILE")
    _add_asr_args(p.add_argument_group("ASR"))
    _add_diar_args(p.add_argument_group("Diarization (same backends as GUI, all local)"))
    _add_output_args(p.add_argument_group("Output"))
    p.add_argument("--project-root", type=Path, default=None, help="Project root path")
    p.add_argument("--models-dir", type=Path, default=None, help="ASR/diarization model cache directory")
    return p


def _add_asr_args(asr: argparse._ArgumentGroup) -> None:
    asr.add_argument("--profile", default="Quality", choices=list(PROFILES), help="ASR quality preset (default: Quality)")
    asr.add_argument("--model", default="large-v3", help="Whisper model: tiny/base/small/medium/large-v2/large-v3")
    asr.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    asr.add_argument("--language", default=None, metavar="LANG", help="Language code: ru, en, de, ...")
    asr.add_argument("--initial-prompt", default=None, metavar="TEXT", help="Domain hint to improve terminology")
    asr.add_argument("--no-vad", action="store_true", help="Disable VAD filter")
    asr.add_argument("--compute-type", default=None, choices=["float16", "int8_float16", "int8", "int8_float32", "float32"])
    asr.add_argument("--beam-size", type=int, default=None)
    asr.add_argument("--cpu-threads", type=int, default=None, help="One-time WhisperModel cpu_threads override")
    asr.add_argument("--num-workers", type=int, default=None, help="One-time WhisperModel num_workers override")
    asr.add_argument("--temperature", type=float, default=None, help="One-time faster-whisper temperature override")
    asr.add_argument("--no-condition-on-previous-text", action="store_true")
    asr.add_argument("--asr-option", action="append", default=[], metavar="KEY=VALUE")


def _add_diar_args(diar: argparse._ArgumentGroup) -> None:
    diar.add_argument("--diar", action="store_true", help="Enable speaker diarization")
    diar.add_argument("--diar-backend", default="online", choices=["online", "pyannote", "nemo", "sherpa_onnx"])
    diar.add_argument("--diar-threshold", type=float, default=0.74, metavar="T")
    diar.add_argument("--diar-device", default=None, help="Device for diarizer (default: same as --device)")
    diar.add_argument("--sherpa-model-path", default="", metavar="PATH")
    diar.add_argument("--sherpa-provider", default="cpu", choices=["cpu", "cuda", "coreml"])
    diar.add_argument("--sherpa-threads", type=int, default=1)


def _add_output_args(out: argparse._ArgumentGroup) -> None:
    out.add_argument("--format", default="srt", choices=list(OUTPUT_FORMATS))
    out.add_argument("--word-by-word", action="store_true")
    out.add_argument("--out-dir", type=Path, default=None, help="Output directory")


def main(argv=None, *, bootstrap=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.project_root or args.models_dir) and bootstrap is not None:
        bootstrap(str(args.project_root) if args.project_root else None, str(args.models_dir) if args.models_dir else None)

    try:
        asr_options = _parse_asr_options(args.asr_option)
    except ValueError as exc:
        parser.error(str(exc))

    files = [f for f in (Path(x) for x in args.files) if f.exists() or print(f"[SKIP] not found: {f}", file=sys.stderr) or False]
    if not files:
        print("No valid input files.", file=sys.stderr)
        sys.exit(1)

    requests = [BatchScribeRequest(input_path=src, **_common_request_kwargs(args, asr_options)) for src in files]
    profile = profile_from_request(requests[0])
    with tempfile.TemporaryDirectory() as tmp:
        with Scribe(profile, **scribe_kwargs_from_request(requests[0])) as scribe:
            _run_batch_requests(scribe, Path(tmp), requests)


def _common_request_kwargs(args, asr_options: Dict[str, Any]) -> dict:  # noqa: ANN001
    return {
        "output_format": args.format,
        "project_root": args.project_root,
        "models_dir": args.models_dir,
        "out_dir": args.out_dir,
        "profile_name": args.profile,
        "model": args.model,
        "device": args.device,
        "language": args.language,
        "initial_prompt": args.initial_prompt,
        "vad_filter": not args.no_vad,
        "condition_on_previous_text": not args.no_condition_on_previous_text,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "cpu_threads": args.cpu_threads,
        "num_workers": args.num_workers,
        "temperature": args.temperature,
        "asr_options": asr_options,
        "word_by_word": args.word_by_word,
        "diar": args.diar,
        "diar_backend": args.diar_backend,
        "diar_threshold": args.diar_threshold,
        "diar_device": args.diar_device,
        "sherpa_model_path": args.sherpa_model_path,
        "sherpa_provider": args.sherpa_provider,
        "sherpa_threads": args.sherpa_threads,
    }


def _run_batch_requests(scribe: Scribe, tmp_dir: Path, requests: list[BatchScribeRequest]) -> None:
    for request in requests:
        src = Path(request.input_path)
        print(f"[{src.name}] transcribing...", end=" ", flush=True)
        try:
            segments = scribe.process(src, tmp_dir, word_by_word=request.word_by_word)
            output_path = request.resolved_output_path()
            output_format = str(request.output_format or "srt").lower()
            write_output(segments, output_path, output_format)
            print(f"-> {output_path}  ({len(segments)} segments)")
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
