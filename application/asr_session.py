from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from asr.domain import Mode, OverloadStrategy
from asr.pipeline import ASRPipeline


class ASRRuntime(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...


@dataclass(frozen=True)
class ASRSessionSettings:
    language: str
    mode: Mode
    model_name: str
    device: str
    compute_type: str
    beam_size: int
    endpoint_silence_ms: float
    max_segment_s: float
    overlap_ms: float
    vad_energy_threshold: float
    overload_strategy: OverloadStrategy
    overload_enter_qsize: int
    overload_exit_qsize: int
    overload_hard_qsize: int
    overload_beam_cap: int
    overload_max_segment_s: float
    overload_overlap_ms: float
    asr_language: Optional[str]
    asr_initial_prompt: Optional[str]


def build_asr_runtime(
    settings: ASRSessionSettings,
    *,
    tap_queue,
    project_root: Path,
    event_queue=None,
) -> ASRRuntime:
    return ASRPipeline(
        tap_queue=tap_queue,
        project_root=project_root,
        language=settings.language,
        mode=settings.mode,
        asr_model_name=settings.model_name,
        device=settings.device,
        compute_type=settings.compute_type,
        beam_size=settings.beam_size,
        endpoint_silence_ms=settings.endpoint_silence_ms,
        max_segment_s=settings.max_segment_s,
        overlap_ms=settings.overlap_ms,
        vad_energy_threshold=settings.vad_energy_threshold,
        vad_hangover_ms=350,
        vad_min_speech_ms=350,
        diarization_enabled=False,
        log_speaker_labels=False,
        overload_strategy=settings.overload_strategy,
        overload_enter_qsize=settings.overload_enter_qsize,
        overload_exit_qsize=settings.overload_exit_qsize,
        overload_hard_drop_qsize=settings.overload_hard_qsize,
        overload_hold_s=2.5,
        overload_beam_cap=settings.overload_beam_cap,
        overload_overlap_ms=settings.overload_overlap_ms,
        overload_max_segment_s=settings.overload_max_segment_s,
        utterance_enabled=True,
        utterance_gap_s=0.85,
        utterance_max_s=18.0,
        utterance_flush_s=2.5,
        log_max_bytes=25 * 1024 * 1024,
        log_backup_count=5,
        event_queue=event_queue,
        asr_language=settings.asr_language,
        asr_initial_prompt=settings.asr_initial_prompt,
        metrics_emit_interval_s=1.0,
        metrics_latency_window=200,
    )
