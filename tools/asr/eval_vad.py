from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
src_text = str(SRC_ROOT)
if src_text not in sys.path:
    sys.path.insert(0, src_text)

from asr.infrastructure.vad import EnergyVAD


@dataclass
class Scenario:
    name: str
    frames: List[np.ndarray]
    labels: List[bool]


@dataclass
class ConfusionMetrics:
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    accuracy: float


@dataclass
class SegmentMetrics:
    total_segments: int
    detected_segments: int
    segment_recall: float
    missed_segments: int
    onset_delay_ms_avg: Optional[float]
    onset_delay_ms_p95: Optional[float]
    release_tail_ms_avg: Optional[float]
    release_tail_ms_p95: Optional[float]
    early_cut_ms_avg: Optional[float]
    early_cut_ms_p95: Optional[float]


@dataclass
class ScenarioResult:
    name: str
    duration_s: float
    raw: ConfusionMetrics
    tolerant: ConfusionMetrics
    segments: SegmentMetrics
    false_positive_ms: float
    false_negative_ms: float


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def _sine(freq_hz: float, t: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * float(freq_hz) * t)


class _ScenarioBuilder:
    def __init__(self, *, sample_rate: int, frame_ms: int, seed: int) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_len = int(round(self.sample_rate * self.frame_ms / 1000.0))
        self._rng = np.random.default_rng(seed)
        self._sample_cursor = 0
        self.frames: List[np.ndarray] = []
        self.labels: List[bool] = []

    def _time_axis(self) -> np.ndarray:
        idx = np.arange(self.frame_len, dtype=np.float32) + float(self._sample_cursor)
        return idx / float(self.sample_rate)

    def _push(self, frame: np.ndarray, *, speech: bool) -> None:
        self.frames.append(np.asarray(frame, dtype=np.float32).reshape(-1))
        self.labels.append(bool(speech))
        self._sample_cursor += self.frame_len

    def add_silence(self, duration_s: float) -> None:
        count = int(round(float(duration_s) * 1000.0 / float(self.frame_ms)))
        for _ in range(count):
            self._push(np.zeros((self.frame_len,), dtype=np.float32), speech=False)

    def add_high_freq_noise(self, duration_s: float, *, amplitude: float = 0.004, freq_hz: float = 6000.0) -> None:
        count = int(round(float(duration_s) * 1000.0 / float(self.frame_ms)))
        for _ in range(count):
            t = self._time_axis()
            phase_jitter = 0.03 * self._rng.standard_normal(self.frame_len)
            carrier = _sine(freq_hz, t + phase_jitter.astype(np.float32))
            hiss = 0.12 * self._rng.standard_normal(self.frame_len)
            frame = float(amplitude) * (carrier + hiss)
            self._push(np.clip(frame, -1.0, 1.0).astype(np.float32), speech=False)

    def add_voiced_speech(
        self,
        duration_s: float,
        *,
        amplitude: float = 0.05,
        f0_hz: float = 220.0,
        bg_noise_amp: float = 0.0,
    ) -> None:
        count = int(round(float(duration_s) * 1000.0 / float(self.frame_ms)))
        for _ in range(count):
            t = self._time_axis()
            envelope = 0.72 + 0.28 * _sine(2.4, t)
            speech = (
                0.55 * _sine(f0_hz, t)
                + 0.28 * _sine(f0_hz * 2.0, t)
                + 0.14 * _sine(f0_hz * 3.0, t)
                + 0.24 * _sine(700.0, t)
                + 0.11 * _sine(1450.0, t)
            )
            speech = speech / 1.32
            breath = 0.02 * self._rng.standard_normal(self.frame_len)
            hiss = np.zeros((self.frame_len,), dtype=np.float32)
            if bg_noise_amp > 0.0:
                hiss_t = self._time_axis()
                hiss = (
                    float(bg_noise_amp)
                    * (_sine(6000.0, hiss_t) + 0.15 * self._rng.standard_normal(self.frame_len))
                ).astype(np.float32)
            frame = float(amplitude) * envelope * (speech + breath) + hiss
            self._push(np.clip(frame, -1.0, 1.0).astype(np.float32), speech=True)


def _build_scenarios(*, sample_rate: int, frame_ms: int) -> List[Scenario]:
    scenarios: List[Scenario] = []

    b = _ScenarioBuilder(sample_rate=sample_rate, frame_ms=frame_ms, seed=101)
    b.add_silence(3.0)
    scenarios.append(Scenario(name="silence_only", frames=b.frames, labels=b.labels))

    b = _ScenarioBuilder(sample_rate=sample_rate, frame_ms=frame_ms, seed=102)
    b.add_high_freq_noise(3.0, amplitude=0.0045)
    scenarios.append(Scenario(name="high_freq_noise", frames=b.frames, labels=b.labels))

    b = _ScenarioBuilder(sample_rate=sample_rate, frame_ms=frame_ms, seed=103)
    b.add_silence(0.8)
    b.add_voiced_speech(1.2, amplitude=0.055, f0_hz=220.0)
    b.add_silence(1.0)
    b.add_voiced_speech(1.0, amplitude=0.050, f0_hz=260.0)
    b.add_silence(0.9)
    scenarios.append(Scenario(name="clear_speech_bursts", frames=b.frames, labels=b.labels))

    b = _ScenarioBuilder(sample_rate=sample_rate, frame_ms=frame_ms, seed=104)
    b.add_silence(1.0)
    b.add_voiced_speech(1.2, amplitude=0.028, f0_hz=240.0)
    b.add_silence(1.0)
    scenarios.append(Scenario(name="near_threshold_speech", frames=b.frames, labels=b.labels))

    b = _ScenarioBuilder(sample_rate=sample_rate, frame_ms=frame_ms, seed=105)
    b.add_high_freq_noise(2.2, amplitude=0.0040)
    b.add_voiced_speech(1.1, amplitude=0.050, f0_hz=220.0, bg_noise_amp=0.0012)
    b.add_silence(0.9)
    scenarios.append(Scenario(name="noise_then_speech", frames=b.frames, labels=b.labels))

    b = _ScenarioBuilder(sample_rate=sample_rate, frame_ms=frame_ms, seed=106)
    b.add_silence(0.6)
    b.add_voiced_speech(0.9, amplitude=0.042, f0_hz=210.0, bg_noise_amp=0.0015)
    b.add_silence(0.9)
    b.add_voiced_speech(0.8, amplitude=0.040, f0_hz=260.0, bg_noise_amp=0.0015)
    b.add_silence(0.8)
    scenarios.append(Scenario(name="noisy_meeting_like", frames=b.frames, labels=b.labels))

    return scenarios


def _find_segments(labels: Sequence[bool]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, val in enumerate(labels):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(labels)))
    return segments


def _confusion(labels: Sequence[bool], preds: Sequence[bool], *, mask: Optional[Sequence[bool]] = None) -> ConfusionMetrics:
    tp = fp = fn = tn = 0
    for idx, (truth, pred) in enumerate(zip(labels, preds)):
        if mask is not None and not mask[idx]:
            continue
        if truth and pred:
            tp += 1
        elif truth and not pred:
            fn += 1
        elif not truth and pred:
            fp += 1
        else:
            tn += 1

    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    accuracy = _safe_ratio(tp + tn, tp + fp + fn + tn)
    return ConfusionMetrics(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def _boundary_mask(labels: Sequence[bool], *, tolerance_frames: int) -> List[bool]:
    keep = [True] * len(labels)
    if tolerance_frames <= 0:
        return keep
    prev = labels[0] if labels else False
    for idx, val in enumerate(labels[1:], start=1):
        if val != prev:
            lo = max(0, idx - tolerance_frames)
            hi = min(len(labels), idx + tolerance_frames)
            for pos in range(lo, hi):
                keep[pos] = False
        prev = val
    return keep


def _segment_metrics(labels: Sequence[bool], preds: Sequence[bool], *, frame_ms: int) -> SegmentMetrics:
    true_segments = _find_segments(labels)
    if not true_segments:
        return SegmentMetrics(
            total_segments=0,
            detected_segments=0,
            segment_recall=0.0,
            missed_segments=0,
            onset_delay_ms_avg=None,
            onset_delay_ms_p95=None,
            release_tail_ms_avg=None,
            release_tail_ms_p95=None,
            early_cut_ms_avg=None,
            early_cut_ms_p95=None,
        )

    onset_delays: List[float] = []
    release_tails: List[float] = []
    early_cuts: List[float] = []
    detected = 0

    for idx, (start, end) in enumerate(true_segments):
        next_start = true_segments[idx + 1][0] if idx + 1 < len(true_segments) else len(labels)
        overlap = any(preds[pos] for pos in range(start, end))
        if overlap:
            detected += 1
            first_pred = next((pos for pos in range(start, end) if preds[pos]), None)
            if first_pred is not None:
                onset_delays.append(float((first_pred - start) * frame_ms))

            last_pred_in_seg = next((pos for pos in range(end - 1, start - 1, -1) if preds[pos]), None)
            if last_pred_in_seg is not None and last_pred_in_seg < (end - 1):
                early_cuts.append(float((end - 1 - last_pred_in_seg) * frame_ms))
            else:
                early_cuts.append(0.0)

            tail_frames = 0
            scan = end
            while scan < next_start and preds[scan]:
                tail_frames += 1
                scan += 1
            release_tails.append(float(tail_frames * frame_ms))

    total_segments = len(true_segments)
    return SegmentMetrics(
        total_segments=total_segments,
        detected_segments=detected,
        segment_recall=_safe_ratio(detected, total_segments),
        missed_segments=total_segments - detected,
        onset_delay_ms_avg=(float(np.mean(onset_delays)) if onset_delays else None),
        onset_delay_ms_p95=_percentile(onset_delays, 95.0),
        release_tail_ms_avg=(float(np.mean(release_tails)) if release_tails else None),
        release_tail_ms_p95=_percentile(release_tails, 95.0),
        early_cut_ms_avg=(float(np.mean(early_cuts)) if early_cuts else None),
        early_cut_ms_p95=_percentile(early_cuts, 95.0),
    )


def _evaluate_scenario(
    scenario: Scenario,
    *,
    vad_factory,
    frame_ms: int,
    boundary_tolerance_ms: int,
) -> ScenarioResult:
    vad = vad_factory()
    preds = [bool(vad.is_speech_frame(frame)) for frame in scenario.frames]

    tol_frames = int(round(float(boundary_tolerance_ms) / float(frame_ms)))
    mask = _boundary_mask(scenario.labels, tolerance_frames=tol_frames)

    raw = _confusion(scenario.labels, preds)
    tolerant = _confusion(scenario.labels, preds, mask=mask)
    seg = _segment_metrics(scenario.labels, preds, frame_ms=frame_ms)

    fp_frames = sum(1 for truth, pred in zip(scenario.labels, preds) if (not truth) and pred)
    fn_frames = sum(1 for truth, pred in zip(scenario.labels, preds) if truth and (not pred))

    return ScenarioResult(
        name=scenario.name,
        duration_s=float(len(scenario.frames) * frame_ms) / 1000.0,
        raw=raw,
        tolerant=tolerant,
        segments=seg,
        false_positive_ms=float(fp_frames * frame_ms),
        false_negative_ms=float(fn_frames * frame_ms),
    )


def _aggregate_confusion(metrics: Iterable[ConfusionMetrics]) -> ConfusionMetrics:
    tp = fp = fn = tn = 0
    for metric in metrics:
        tp += int(metric.tp)
        fp += int(metric.fp)
        fn += int(metric.fn)
        tn += int(metric.tn)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall)
    accuracy = _safe_ratio(tp + tn, tp + fp + fn + tn)
    return ConfusionMetrics(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall, f1=f1, accuracy=accuracy)


def _benchmark(scenarios: Sequence[Scenario], *, vad_factory, frame_ms: int, loops: int, warmup_loops: int) -> Dict[str, float]:
    corpus = [frame for scenario in scenarios for frame in scenario.frames]
    corpus_frames = len(corpus)
    corpus_audio_s = float(corpus_frames * frame_ms) / 1000.0

    for _ in range(max(0, warmup_loops)):
        vad = vad_factory()
        for frame in corpus:
            vad.is_speech_frame(frame)

    t0 = time.perf_counter()
    for _ in range(max(1, loops)):
        vad = vad_factory()
        for frame in corpus:
            vad.is_speech_frame(frame)
    elapsed_s = time.perf_counter() - t0

    total_frames = corpus_frames * max(1, loops)
    total_audio_s = corpus_audio_s * max(1, loops)
    ms_per_frame = (elapsed_s * 1000.0) / float(max(1, total_frames))
    x_realtime = _safe_ratio(total_audio_s, elapsed_s)
    realtime_factor = _safe_ratio(elapsed_s, total_audio_s)

    return {
        "loops": int(max(1, loops)),
        "corpus_frames": int(corpus_frames),
        "corpus_audio_s": float(corpus_audio_s),
        "elapsed_s": float(elapsed_s),
        "total_audio_s": float(total_audio_s),
        "ms_per_frame": float(ms_per_frame),
        "x_realtime": float(x_realtime),
        "realtime_factor": float(realtime_factor),
    }


def _build_vad_factory(args: argparse.Namespace):
    def factory() -> EnergyVAD:
        return EnergyVAD(
            sample_rate=int(args.sample_rate),
            frame_ms=int(args.frame_ms),
            energy_threshold=float(args.energy_threshold),
            adaptive=bool(args.adaptive),
            noise_mult=float(args.noise_mult),
            noise_alpha=float(args.noise_alpha),
            hangover_ms=int(args.hangover_ms),
            min_speech_ms=int(args.min_speech_ms),
            band_ratio_min=float(args.band_ratio_min),
            voiced_min=float(args.voiced_min),
            band_ratio_weight=float(args.band_ratio_weight),
            voiced_weight=float(args.voiced_weight),
            pre_speech_ms=int(args.pre_speech_ms),
            min_end_silence_ms=int(args.min_end_silence_ms),
            voiced_every_n_frames=int(args.voiced_every_n_frames),
            voiced_only_near_thr=bool(args.voiced_only_near_thr),
            near_thr_ratio=float(args.near_thr_ratio),
        )

    return factory


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic quality + performance eval for EnergyVAD.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--energy-threshold", type=float, default=0.006)
    parser.add_argument("--adaptive", dest="adaptive", action="store_true", default=True)
    parser.add_argument("--no-adaptive", dest="adaptive", action="store_false")
    parser.add_argument("--noise-mult", type=float, default=3.0)
    parser.add_argument("--noise-alpha", type=float, default=0.05)
    parser.add_argument("--hangover-ms", type=int, default=400)
    parser.add_argument("--min-speech-ms", type=int, default=350)
    parser.add_argument("--band-ratio-min", type=float, default=0.35)
    parser.add_argument("--voiced-min", type=float, default=0.12)
    parser.add_argument("--band-ratio-weight", type=float, default=0.55)
    parser.add_argument("--voiced-weight", type=float, default=0.45)
    parser.add_argument("--pre-speech-ms", type=int, default=120)
    parser.add_argument("--min-end-silence-ms", type=int, default=220)
    parser.add_argument("--voiced-every-n-frames", type=int, default=2)
    parser.add_argument("--voiced-only-near-thr", dest="voiced_only_near_thr", action="store_true", default=True)
    parser.add_argument("--always-compute-voiced", dest="voiced_only_near_thr", action="store_false")
    parser.add_argument("--near-thr-ratio", type=float, default=0.70)
    parser.add_argument("--boundary-tolerance-ms", type=int, default=400)
    parser.add_argument("--benchmark-loops", type=int, default=100)
    parser.add_argument("--warmup-loops", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-tolerant-f1", type=float, default=None)
    parser.add_argument("--min-segment-recall", type=float, default=None)
    parser.add_argument("--min-x-realtime", type=float, default=None)
    parser.add_argument("--max-ms-per-frame", type=float, default=None)
    return parser


def _threshold_failures(report: Dict[str, object], args: argparse.Namespace) -> List[str]:
    failures: List[str] = []
    summary = report["summary"]
    benchmark = report["benchmark"]

    tolerant_f1 = float(summary["tolerant"]["f1"])
    segment_recall = float(summary["segments"]["segment_recall"])
    x_realtime = float(benchmark["x_realtime"])
    ms_per_frame = float(benchmark["ms_per_frame"])

    if args.min_tolerant_f1 is not None and tolerant_f1 < float(args.min_tolerant_f1):
        failures.append(f"tolerant_f1 {tolerant_f1:.4f} < {float(args.min_tolerant_f1):.4f}")
    if args.min_segment_recall is not None and segment_recall < float(args.min_segment_recall):
        failures.append(f"segment_recall {segment_recall:.4f} < {float(args.min_segment_recall):.4f}")
    if args.min_x_realtime is not None and x_realtime < float(args.min_x_realtime):
        failures.append(f"x_realtime {x_realtime:.2f} < {float(args.min_x_realtime):.2f}")
    if args.max_ms_per_frame is not None and ms_per_frame > float(args.max_ms_per_frame):
        failures.append(f"ms_per_frame {ms_per_frame:.6f} > {float(args.max_ms_per_frame):.6f}")

    return failures


def _format_report(report: Dict[str, object]) -> str:
    lines: List[str] = []
    summary = report["summary"]
    benchmark = report["benchmark"]
    scenarios = report["scenarios"]

    lines.append("EnergyVAD synthetic eval")
    lines.append(
        f"raw_f1={summary['raw']['f1']:.4f} tolerant_f1={summary['tolerant']['f1']:.4f} "
        f"segment_recall={summary['segments']['segment_recall']:.4f}"
    )
    lines.append(
        f"onset_avg_ms={summary['segments']['onset_delay_ms_avg']} "
        f"release_tail_avg_ms={summary['segments']['release_tail_ms_avg']} "
        f"early_cut_avg_ms={summary['segments']['early_cut_ms_avg']}"
    )
    lines.append(
        f"ms_per_frame={benchmark['ms_per_frame']:.6f} "
        f"x_realtime={benchmark['x_realtime']:.2f} "
        f"rtf={benchmark['realtime_factor']:.6f}"
    )
    lines.append("")
    lines.append("Per scenario:")
    for scenario in scenarios:
        lines.append(
            f"- {scenario['name']}: raw_f1={scenario['raw']['f1']:.4f} "
            f"tolerant_f1={scenario['tolerant']['f1']:.4f} "
            f"seg_recall={scenario['segments']['segment_recall']:.4f} "
            f"fp_ms={scenario['false_positive_ms']:.1f} fn_ms={scenario['false_negative_ms']:.1f}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    vad_factory = _build_vad_factory(args)
    scenarios = _build_scenarios(sample_rate=int(args.sample_rate), frame_ms=int(args.frame_ms))

    scenario_results = [
        _evaluate_scenario(
            scenario,
            vad_factory=vad_factory,
            frame_ms=int(args.frame_ms),
            boundary_tolerance_ms=int(args.boundary_tolerance_ms),
        )
        for scenario in scenarios
    ]

    raw_summary = _aggregate_confusion(result.raw for result in scenario_results)
    tolerant_summary = _aggregate_confusion(result.tolerant for result in scenario_results)

    segment_totals = [result.segments for result in scenario_results if result.segments.total_segments > 0]
    onset_vals = [value for seg in segment_totals for value in ([seg.onset_delay_ms_avg] if seg.onset_delay_ms_avg is not None else [])]
    release_vals = [value for seg in segment_totals for value in ([seg.release_tail_ms_avg] if seg.release_tail_ms_avg is not None else [])]
    early_cut_vals = [value for seg in segment_totals for value in ([seg.early_cut_ms_avg] if seg.early_cut_ms_avg is not None else [])]

    detected_segments = sum(seg.detected_segments for seg in segment_totals)
    total_segments = sum(seg.total_segments for seg in segment_totals)
    segment_summary = {
        "total_segments": int(total_segments),
        "detected_segments": int(detected_segments),
        "segment_recall": _safe_ratio(detected_segments, total_segments),
        "missed_segments": int(total_segments - detected_segments),
        "onset_delay_ms_avg": (float(np.mean(onset_vals)) if onset_vals else None),
        "onset_delay_ms_p95": _percentile(onset_vals, 95.0),
        "release_tail_ms_avg": (float(np.mean(release_vals)) if release_vals else None),
        "release_tail_ms_p95": _percentile(release_vals, 95.0),
        "early_cut_ms_avg": (float(np.mean(early_cut_vals)) if early_cut_vals else None),
        "early_cut_ms_p95": _percentile(early_cut_vals, 95.0),
    }

    benchmark = _benchmark(
        scenarios,
        vad_factory=vad_factory,
        frame_ms=int(args.frame_ms),
        loops=int(args.benchmark_loops),
        warmup_loops=int(args.warmup_loops),
    )

    report: Dict[str, object] = {
        "config": {
            "sample_rate": int(args.sample_rate),
            "frame_ms": int(args.frame_ms),
            "energy_threshold": float(args.energy_threshold),
            "adaptive": bool(args.adaptive),
            "noise_mult": float(args.noise_mult),
            "noise_alpha": float(args.noise_alpha),
            "hangover_ms": int(args.hangover_ms),
            "min_speech_ms": int(args.min_speech_ms),
            "band_ratio_min": float(args.band_ratio_min),
            "voiced_min": float(args.voiced_min),
            "band_ratio_weight": float(args.band_ratio_weight),
            "voiced_weight": float(args.voiced_weight),
            "pre_speech_ms": int(args.pre_speech_ms),
            "min_end_silence_ms": int(args.min_end_silence_ms),
            "voiced_every_n_frames": int(args.voiced_every_n_frames),
            "voiced_only_near_thr": bool(args.voiced_only_near_thr),
            "near_thr_ratio": float(args.near_thr_ratio),
            "boundary_tolerance_ms": int(args.boundary_tolerance_ms),
        },
        "summary": {
            "raw": asdict(raw_summary),
            "tolerant": asdict(tolerant_summary),
            "segments": segment_summary,
        },
        "benchmark": benchmark,
        "scenarios": [asdict(result) for result in scenario_results],
    }

    failures = _threshold_failures(report, args)
    if failures:
        report["failures"] = failures

    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print(_format_report(report))
        if failures:
            print("")
            print("Threshold failures:")
            for failure in failures:
                print(f"- {failure}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
