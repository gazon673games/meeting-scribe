#!/usr/bin/env python3
"""
ASR latency benchmark.

Usage:
    python -m tests.bench_asr_latency
    python -m tests.bench_asr_latency --device cpu --runs 3
    python -m tests.bench_asr_latency --device cuda --runs 5 --warmup 2
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

SAMPLE_RATE = 16_000
SEGMENT_DURATIONS_S = [3.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Configs to compare
# ---------------------------------------------------------------------------

@dataclass
class BenchConfig:
    label: str
    model: str
    compute_type: str
    beam_size: int
    device: str
    language: str = "ru"


def default_configs(device: str) -> List[BenchConfig]:
    return [
        BenchConfig(
            label="large-v3 float16 beam=6  [current]",
            model="large-v3",
            compute_type="float16",
            beam_size=6,
            device=device,
        ),
        BenchConfig(
            label="large-v3-russian float16 beam=6",
            model="bzikst/faster-whisper-large-v3-russian",
            compute_type="float16",
            beam_size=6,
            device=device,
        ),
        BenchConfig(
            label="large-v3-turbo int8 beam=5",
            model="large-v3-turbo",
            compute_type="int8_float16",
            beam_size=5,
            device=device,
        ),
        BenchConfig(
            label="large-v3-turbo int8 beam=3  [fastest]",
            model="large-v3-turbo",
            compute_type="int8_float16",
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


def _load_wav(path: Path) -> Optional[np.ndarray]:
    try:
        import wave, struct
        with wave.open(str(path), "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        if sr != SAMPLE_RATE:
            factor = SAMPLE_RATE / sr
            new_len = int(len(samples) * factor)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, new_len),
                np.arange(len(samples)),
                samples,
            ).astype(np.float32)
        return samples
    except Exception as e:
        print(f"  [warn] could not load {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    config_label: str
    duration_s: float
    latencies_s: List[float] = field(default_factory=list)

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
        return s[max(0, int(len(s) * 0.95) - 1)]

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.latencies_s) if len(self.latencies_s) > 1 else 0.0

    @property
    def rtf(self) -> float:
        return self.mean / self.duration_s if self.duration_s > 0 else 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _bench_config(
    cfg: BenchConfig,
    audios: List[np.ndarray],
    durations: List[float],
    warmup: int,
    runs: int,
) -> List[RunResult]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from asr.infrastructure.worker_faster_whisper import FasterWhisperASR

    print(f"\n{'-'*64}")
    print(f"  {cfg.label}")
    print(f"  model={cfg.model}  compute={cfg.compute_type}  beam={cfg.beam_size}  device={cfg.device}")

    print("  loading model...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        asr = FasterWhisperASR(
            model_name=cfg.model,
            language=cfg.language,
            device=cfg.device,
            compute_type=cfg.compute_type,
            beam_size=cfg.beam_size,
        )
    except Exception as e:
        print(f"FAILED: {e}")
        return []
    print(f"ok ({time.perf_counter() - t0:.1f}s)")

    results: List[RunResult] = []

    for audio, dur in zip(audios, durations):
        result = RunResult(config_label=cfg.label, duration_s=dur)

        for _ in range(warmup):
            asr.transcribe(audio, beam_size=cfg.beam_size)

        for _ in range(runs):
            t_start = time.perf_counter()
            asr.transcribe(audio, beam_size=cfg.beam_size)
            result.latencies_s.append(time.perf_counter() - t_start)

        results.append(result)
        print(
            f"  {dur:4.0f}s seg | "
            f"mean={result.mean*1000:5.0f}ms  "
            f"p50={result.median*1000:5.0f}ms  "
            f"p95={result.p95*1000:5.0f}ms  "
            f"rtf={result.rtf:.3f}"
        )

    del asr
    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(all_results: List[RunResult], silence_ms: float) -> None:
    durations = sorted({r.duration_s for r in all_results})
    labels = list(dict.fromkeys(r.config_label for r in all_results))

    def get(label: str, dur: float) -> Optional[RunResult]:
        for r in all_results:
            if r.config_label == label and r.duration_s == dur:
                return r
        return None

    print(f"\n{'='*64}")
    print("  SUMMARY -- inference latency (ms)")
    print(f"{'='*64}")

    col = 16
    header = f"  {'config':<28}" + "".join(f"  {d:.0f}s seg".rjust(col) for d in durations)
    print(header)
    print(f"  {'-'*28}" + "-" * (col + 2) * len(durations))

    baseline: dict[float, float] = {}
    for i, label in enumerate(labels):
        row = f"  {label:<28}"
        for dur in durations:
            r = get(label, dur)
            if r is None:
                row += f"  {'n/a':>{col-2}}"
            else:
                ms = r.mean * 1000
                if i == 0:
                    baseline[dur] = ms
                    row += f"  {ms:>8.0f} ms      "
                else:
                    delta = ms - baseline.get(dur, ms)
                    sign = "+" if delta >= 0 else ""
                    row += f"  {ms:>8.0f} ms {sign}{delta:+.0f}"
        print(row)

    print(f"\n  NOTE: add ~{silence_ms:.0f}ms endpoint_silence for total perceived latency")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ASR latency benchmark")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--runs", type=int, default=5, help="timed runs per segment")
    parser.add_argument("--warmup", type=int, default=2, help="warmup runs (not timed)")
    parser.add_argument("--silence-ms", type=float, default=500.0,
                        help="endpoint_silence_ms from config (for summary note)")
    parser.add_argument("--wav", type=str, default=None,
                        help="path to a WAV file to use instead of synthetic audio")
    parser.add_argument("--configs", choices=["all", "quick"], default="all",
                        help="'all' runs 3 configs, 'quick' runs only baseline+optimized")
    args = parser.parse_args()

    print(f"\nASR Latency Benchmark")
    print(f"device={args.device}  runs={args.runs}  warmup={args.warmup}")

    configs = default_configs(args.device)
    if args.configs == "quick":
        configs = configs[:2]

    # Build audio clips
    if args.wav:
        wav_path = Path(args.wav)
        full = _load_wav(wav_path)
        if full is None:
            sys.exit(1)
        audios = []
        for dur in SEGMENT_DURATIONS_S:
            n = int(dur * SAMPLE_RATE)
            clip = full[:n] if len(full) >= n else np.pad(full, (0, n - len(full)))
            audios.append(clip.astype(np.float32))
        print(f"audio source: {wav_path.name}")
    else:
        audios = [_make_audio(d) for d in SEGMENT_DURATIONS_S]
        print("audio source: synthetic (voiced sine + noise)")

    all_results: List[RunResult] = []
    for cfg in configs:
        results = _bench_config(cfg, audios, SEGMENT_DURATIONS_S, args.warmup, args.runs)
        all_results.extend(results)

    if all_results:
        _print_summary(all_results, args.silence_ms)


if __name__ == "__main__":
    main()
