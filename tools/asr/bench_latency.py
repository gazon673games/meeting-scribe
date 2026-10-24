#!/usr/bin/env python3
"""
ASR latency benchmark.

Usage:
    python -m tools.asr.bench_latency
    python -m tools.asr.bench_latency --device cpu --runs 3
    python -m tools.asr.bench_latency --device cuda --runs 5 --warmup 2
    python -m tools.asr.bench_latency --configs native --device cpu
    # Add --wav sample.wav --reference sample.txt for WER (up to 30 seconds).
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / 'backend' / 'src'
src_text = str(SRC_ROOT)
if src_text not in sys.path:
    sys.path.insert(0, src_text)

SAMPLE_RATE = 16_000
SEGMENT_DURATIONS_S = [3.0, 5.0, 10.0]

if TYPE_CHECKING:
    from asr.application.ports import AsrBackendPort


# ---------------------------------------------------------------------------
# Configs to compare
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchConfig:
    label: str
    model: str
    compute_type: str
    beam_size: int
    device: str
    language: str = 'ru'


def default_configs(device: str) -> list[BenchConfig]:
    precision = 'float32' if device == 'cpu' else 'float16'
    quantized = 'int8' if device == 'cpu' else 'int8_float16'
    return [
        BenchConfig(
            label=f'large-v3 {precision} beam=6  [baseline]',
            model='large-v3',
            compute_type=precision,
            beam_size=6,
            device=device,
        ),
        BenchConfig(
            label=f'large-v3-russian {precision} beam=6',
            model='bzikst/faster-whisper-large-v3-russian',
            compute_type=precision,
            beam_size=6,
            device=device,
        ),
        BenchConfig(
            label='large-v3-turbo int8 beam=5',
            model='large-v3-turbo',
            compute_type=quantized,
            beam_size=5,
            device=device,
        ),
        BenchConfig(
            label='large-v3-turbo int8 beam=3  [fastest]',
            model='large-v3-turbo',
            compute_type=quantized,
            beam_size=3,
            device=device,
        ),
    ]


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------


def _make_audio(duration_s: float) -> np.ndarray:
    """Synthetic voiced audio: fundamental + harmonics + noise."""
    n = int(duration_s * SAMPLE_RATE)
    t = np.linspace(0, duration_s, n, dtype=np.float32)
    rng = np.random.default_rng(42)
    signal = (
        0.30 * np.sin(2 * np.pi * 150 * t)
        + 0.15 * np.sin(2 * np.pi * 300 * t)
        + 0.08 * np.sin(2 * np.pi * 450 * t)
        + 0.03 * rng.standard_normal(n).astype(np.float32)
    )
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)).astype(np.float32)
    return (signal * envelope).astype(np.float32)


def _load_wav(path: Path) -> np.ndarray | None:
    try:
        import wave

        with wave.open(str(path), 'rb') as wf:
            if wf.getsampwidth() != 2:
                raise ValueError(
                    'Benchmark WAV input must contain PCM16 samples'
                )
            ch = wf.getnchannels()
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
        samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
        if ch > 1:
            samples = samples.reshape(-1, ch).mean(axis=1)
        if sr != SAMPLE_RATE:
            factor = SAMPLE_RATE / sr
            new_len = int(len(samples) * factor)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)
        return samples
    except (OSError, ValueError, EOFError, wave.Error) as error:
        print(f'  [warn] could not load {path}: {error}')
        return None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    config_label: str
    duration_s: float
    latencies_s: list[float] = field(default_factory=list)
    wer: float | None = None

    @property
    def n(self) -> int:
        return len(self.latencies_s)

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_s) if self.latencies_s else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.latencies_s) if self.latencies_s else 0.0

    @property
    def p95(self) -> float:
        if not self.latencies_s:
            return 0.0
        s = sorted(self.latencies_s)
        return s[max(0, math.ceil(len(s) * 0.95) - 1)]

    @property
    def stdev(self) -> float:
        return (
            statistics.stdev(self.latencies_s)
            if len(self.latencies_s) > 1
            else 0.0
        )

    @property
    def rtf(self) -> float:
        return self.mean / self.duration_s if self.duration_s > 0 else 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _create_benchmark_backend(cfg: BenchConfig) -> AsrBackendPort:
    from application.native_asr_models import (
        NEMOTRON_MODEL_ID,
        SHERPA_OFFLINE_MODEL_IDS,
        native_asr_model_dir,
    )
    from asr.infrastructure.worker_faster_whisper import FasterWhisperASR
    from asr.infrastructure.worker_sherpa_nemotron import SherpaNemotronASR
    from asr.infrastructure.worker_sherpa_offline import SherpaOfflineASR

    options = {
        'model_name': cfg.model,
        'language': cfg.language,
        'device': cfg.device,
        'compute_type': cfg.compute_type,
        'beam_size': cfg.beam_size,
        'cpu_threads': 0,
        'num_workers': 1,
        'initial_prompt': None,
        'hotwords': None,
    }
    if cfg.model == NEMOTRON_MODEL_ID or cfg.model in SHERPA_OFFLINE_MODEL_IDS:
        backend_type = (
            SherpaNemotronASR
            if cfg.model == NEMOTRON_MODEL_ID
            else SherpaOfflineASR
        )
        return backend_type(
            model_dir=native_asr_model_dir(
                cfg.model, project_root=PROJECT_ROOT
            ),
            **options,
        )
    return FasterWhisperASR(**options)


def _bench_config(
    cfg: BenchConfig,
    audios: list[np.ndarray],
    durations: list[float],
    warmup: int,
    runs: int,
    *,
    reference: str | None = None,
) -> list[RunResult]:
    print(f'\n{"-" * 64}')
    print(f'  {cfg.label}')
    print(
        f'  model={cfg.model}  compute={cfg.compute_type}  beam={cfg.beam_size}  device={cfg.device}'
    )
    print('  loading model...', end=' ', flush=True)
    started_at = time.perf_counter()
    try:
        asr = _create_benchmark_backend(cfg)
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        print(f'FAILED: {error}')
        return []
    try:
        print(f'ok ({time.perf_counter() - started_at:.1f}s)')
        return _measure_config(
            asr, cfg, audios, durations, warmup, runs, reference=reference
        )
    finally:
        asr.close()


def _measure_config(
    asr: AsrBackendPort,
    cfg: BenchConfig,
    audios: list[np.ndarray],
    durations: list[float],
    warmup: int,
    runs: int,
    *,
    reference: str | None = None,
) -> list[RunResult]:
    results: list[RunResult] = []

    for audio, dur in zip(audios, durations):
        result = RunResult(config_label=cfg.label, duration_s=dur)

        for _ in range(warmup):
            _transcribe_checked(asr, audio, beam_size=cfg.beam_size)

        for _ in range(runs):
            t_start = time.perf_counter()
            text = _transcribe_checked(asr, audio, beam_size=cfg.beam_size)
            result.latencies_s.append(time.perf_counter() - t_start)
        if reference is not None:
            result.wer = word_error_rate(reference, text)

        results.append(result)
        print(
            f'  {dur:4.0f}s seg | '
            f'mean={result.mean * 1000:5.0f}ms  '
            f'p50={result.median * 1000:5.0f}ms  '
            f'p95={result.p95 * 1000:5.0f}ms  '
            f'rtf={result.rtf:.3f}'
        )
        if result.wer is not None:
            print(f'  WER={result.wer:.2%} (case and punctuation normalized)')

    return results


def _transcribe_checked(
    asr: AsrBackendPort, audio: np.ndarray, *, beam_size: int
) -> str:
    result = asr.transcribe(audio, beam_size=beam_size)
    if not isinstance(result, dict):
        raise TypeError('ASR benchmark result must be a mapping')
    if result.get('error'):
        raise RuntimeError(str(result['error']))
    return str(result.get('text') or '')


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = re.findall(r'\w+', reference.casefold())
    actual = re.findall(r'\w+', hypothesis.casefold())
    if not expected:
        raise ValueError('Reference transcript is empty')
    previous = list(range(len(actual) + 1))
    for row, word in enumerate(expected, start=1):
        current = [row]
        for column, recognized in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (word != recognized),
                )
            )
        previous = current
    return previous[-1] / len(expected)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _print_summary(all_results: list[RunResult], silence_ms: float) -> None:
    durations = sorted({r.duration_s for r in all_results})
    labels = list(dict.fromkeys(r.config_label for r in all_results))

    def get(label: str, dur: float) -> RunResult | None:
        for r in all_results:
            if r.config_label == label and r.duration_s == dur:
                return r
        return None

    print(f'\n{"=" * 64}')
    print('  SUMMARY -- inference latency (ms)')
    print(f'{"=" * 64}')

    col = 16
    header = f'  {"config":<28}' + ''.join(
        f'  {d:.0f}s seg'.rjust(col) for d in durations
    )
    print(header)
    print(f'  {"-" * 28}' + '-' * (col + 2) * len(durations))

    baseline: dict[float, float] = {}
    for i, label in enumerate(labels):
        row = f'  {label:<28}'
        for dur in durations:
            r = get(label, dur)
            if r is None:
                row += f'  {"n/a":>{col - 2}}'
            else:
                ms = r.mean * 1000
                if i == 0:
                    baseline[dur] = ms
                    row += f'  {ms:>8.0f} ms      '
                else:
                    delta = ms - baseline.get(dur, ms)
                    row += f'  {ms:>8.0f} ms {delta:+.0f}'
        print(row)

    print(
        f'\n  NOTE: add ~{silence_ms:.0f}ms endpoint_silence for total perceived latency'
    )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description='ASR latency benchmark')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument(
        '--runs', type=int, default=5, help='timed runs per segment'
    )
    parser.add_argument(
        '--warmup', type=int, default=2, help='warmup runs (not timed)'
    )
    parser.add_argument(
        '--silence-ms',
        type=float,
        default=500.0,
        help='endpoint_silence_ms from config (for summary note)',
    )
    parser.add_argument(
        '--wav',
        type=str,
        default=None,
        help='path to a WAV file to use instead of synthetic audio',
    )
    parser.add_argument(
        '--configs',
        choices=['all', 'quick', 'native'],
        default='all',
        help="'all' runs Whisper+ONNX, 'quick' runs two Whisper configs, 'native' runs ONNX",
    )
    parser.add_argument(
        '--reference',
        type=Path,
        help='Reference text for a WAV sample up to 30 seconds',
    )
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error('--runs must be positive and --warmup must be nonnegative')
    if args.reference and not args.wav:
        parser.error('--reference requires --wav')

    print('\nASR Latency Benchmark')
    print(f'device={args.device}  runs={args.runs}  warmup={args.warmup}')

    configs = default_configs(args.device)
    if args.configs == 'quick':
        configs = [configs[0], configs[-1]]
    else:
        from application.native_asr_models import (
            NEMOTRON_MODEL_ID,
            PARAKEET_MODEL_ID,
            QWEN_ASR_MODEL_ID,
        )

        native_configs = [
            BenchConfig(f'{name} CPU INT8', name, 'int8', 1, 'cpu')
            for name in (
                NEMOTRON_MODEL_ID,
                PARAKEET_MODEL_ID,
                QWEN_ASR_MODEL_ID,
            )
        ]
        configs = (
            native_configs
            if args.configs == 'native'
            else configs + native_configs
        )

    # Build audio clips
    reference = None
    durations = SEGMENT_DURATIONS_S
    if args.wav:
        wav_path = Path(args.wav)
        full = _load_wav(wav_path)
        if full is None:
            sys.exit(1)
        audios = []
        for dur in SEGMENT_DURATIONS_S:
            n = int(dur * SAMPLE_RATE)
            clip = (
                full[:n] if len(full) >= n else np.pad(full, (0, n - len(full)))
            )
            audios.append(clip.astype(np.float32))
        print(f'audio source: {wav_path.name}')
        if args.reference:
            if len(full) > 30 * SAMPLE_RATE:
                parser.error(
                    '--reference requires a WAV sample up to 30 seconds'
                )
            reference = args.reference.read_text(encoding='utf-8').strip()
            if not reference:
                parser.error('Reference transcript is empty')
            audios = [full]
            durations = [len(full) / SAMPLE_RATE]
    else:
        audios = [_make_audio(d) for d in SEGMENT_DURATIONS_S]
        print('audio source: synthetic (voiced sine + noise)')

    all_results: list[RunResult] = []
    for cfg in configs:
        results = _bench_config(
            cfg,
            audios,
            durations,
            args.warmup,
            args.runs,
            reference=reference,
        )
        all_results.extend(results)

    if all_results:
        _print_summary(all_results, args.silence_ms)
    else:
        raise RuntimeError('No benchmark configuration completed successfully')


if __name__ == '__main__':
    main()
