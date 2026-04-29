from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from asr.application.events import ASREventPublisher
from asr.application.ingest import TapIngestRuntime
from asr.application.metrics import ASRMetrics
from asr.application.overload import OverloadController
from asr.application.pipeline_config import (
    ASRPipelineDependencies,
    ASRPipelineSettings,
    build_diarization_config,
    build_segmenter_config,
)
from asr.application.policies import AdaptiveBeam
from asr.application.ports import AsrLoggerPort, RealtimeWorkerRunnerPort, StopSignalPort, WorkerHandlePort
from asr.application.segmentation import AudioSegmenterPort, SegmenterConfig
from asr.application.transcription_worker import TranscriptionWorkerRuntime
from asr.application.utterances import UtteranceAggregator
from asr.application.worker_config import TranscriptionWorkerConfig
from asr.domain.segments import Segment
from diarization.application.diarization import DiarizationPort
from diarization.application.diarization_updates import DiarizationUpdateConfig, DiarizationUpdateRuntime


@dataclass
class ASRRuntimeGraph:
    worker_runner: RealtimeWorkerRunnerPort
    stop: StopSignalPort
    logger: AsrLoggerPort
    events: ASREventPublisher
    overload: OverloadController
    diarization: DiarizationPort
    utterances: UtteranceAggregator
    metrics: ASRMetrics
    segmenter_config: SegmenterConfig
    segmenter: AudioSegmenterPort
    worker: TranscriptionWorkerRuntime
    diarization_updates: Optional[DiarizationUpdateRuntime]
    ingest: TapIngestRuntime
    ingest_worker: Optional[WorkerHandlePort] = None
    transcription_worker: Optional[WorkerHandlePort] = None
    diarization_worker: Optional[WorkerHandlePort] = None

    def start(self, *, settings: ASRPipelineSettings, session_id: str) -> None:
        self.stop.clear()
        self.overload.reset()
        self.metrics.reset()
        self.segmenter.reset_runtime()
        self.worker.reset_runtime()
        if self.diarization_updates is not None:
            self.diarization_updates.reset_runtime()

        self.events.log(self._build_started_event(settings=settings, session_id=session_id))
        self.ingest_worker = self.worker_runner.start_worker(name="asr-ingest", target=self.ingest.run_safe)
        self.transcription_worker = self.worker_runner.start_worker(
            name="asr-worker",
            target=self.worker.run_safe,
        )
        if self.diarization_updates is not None:
            self.diarization_worker = self.worker_runner.start_worker(
                name="asr-diarization",
                target=self.diarization_updates.run_safe,
            )

    def stop_runtime(self) -> None:
        self.stop.set()
        if self.ingest_worker:
            self.ingest_worker.join()
            self.ingest_worker = None
        if self.transcription_worker:
            self.transcription_worker.join()
            self.transcription_worker = None
        if self.diarization_worker:
            self.diarization_worker.join()
            self.diarization_worker = None

        try:
            self.worker.flush_utterances(force=True)
        except Exception:
            pass

        self.worker.emit_metrics(force=True)
        self.events.log({"type": "asr_stopped", "ts": time.time()})
        self.logger.close()

    def segmentation_params(self, settings: ASRPipelineSettings) -> Tuple[float, float, float]:
        return self.overload.segmentation_params(
            endpoint_silence_ms=settings.endpoint_silence_ms,
            max_segment_s=settings.max_segment_s,
            overlap_ms=settings.overlap_ms,
        )

    def _build_started_event(self, *, settings: ASRPipelineSettings, session_id: str) -> dict:
        return {
            "type": "asr_started",
            "session_id": session_id,
            "language": settings.language,
            "mode": settings.mode,
            "model": settings.asr_model_name,
            "device": settings.device,
            "compute_type": settings.compute_type,
            "cpu_threads": int(settings.cpu_threads),
            "num_workers": int(settings.num_workers),
            "beam_size": int(settings.beam_size),
            "endpoint_silence_ms": settings.endpoint_silence_ms,
            "max_segment_s": settings.max_segment_s,
            "overlap_ms": settings.overlap_ms,
            "overload_strategy": self.overload.strategy,
            "vad": self.segmenter_config.to_event_dict(),
            "overload": self.overload.to_event_dict(),
            "diarization_enabled": self.diarization.enabled,
            "diar_backend": self.diarization.backend,
            "agc_enabled": self.segmenter_config.agc_enabled,
            "text_dedup_enabled": settings.text_dedup_enabled,
            "adaptive_beam_enabled": settings.adaptive_beam_enabled,
            **self.utterances.to_event_dict(),
            "log_rotation": {"max_bytes": self.logger.max_bytes, "backup_count": self.logger.backup_count},
            "log_speaker_labels": settings.log_speaker_labels,
            "asr_language": settings.asr_language,
            "asr_initial_prompt": bool(settings.asr_initial_prompt),
            "ts": time.time(),
        }


def build_runtime_graph(
    *,
    settings: ASRPipelineSettings,
    dependencies: ASRPipelineDependencies,
    tap_queue: "queue.Queue[dict]",
    project_root: Any,
    session_id: str,
    ui_queue: Optional["queue.Queue[dict]"] = None,
    event_queue: Optional["queue.Queue[dict]"] = None,
) -> ASRRuntimeGraph:
    stop = dependencies.worker_runner.create_stop_signal()
    segment_queue: "queue.Queue[Segment]" = queue.Queue(maxsize=50)
    diarization_queue = _build_diarization_queue(settings)

    logger = dependencies.logger_factory(
        root=project_root,
        session_id=session_id,
        language=settings.language,
        max_bytes=int(settings.log_max_bytes),
        backup_count=int(settings.log_backup_count),
    )
    events = ASREventPublisher(
        logger=logger,
        event_queue=event_queue if event_queue is not None else ui_queue,
    )

    overload = OverloadController.from_settings(settings)
    beam_controller = AdaptiveBeam.from_settings(settings)
    utterances = UtteranceAggregator.from_settings(settings)
    metrics = ASRMetrics.from_settings(settings)
    segmenter_config = build_segmenter_config(settings)

    diarization = dependencies.diarization_factory(
        config=build_diarization_config(settings, project_root=project_root)
    )

    def segmentation_params() -> Tuple[float, float, float]:
        return overload.segmentation_params(
            endpoint_silence_ms=settings.endpoint_silence_ms,
            max_segment_s=settings.max_segment_s,
            overlap_ms=settings.overlap_ms,
        )

    segmenter = dependencies.segmenter_factory(
        config=segmenter_config,
        segment_queue=segment_queue,
        diarization=diarization,
        metrics=metrics,
        log_event=events.log,
        segmentation_params=segmentation_params,
        diarization_queue=diarization_queue,
    )
    worker = TranscriptionWorkerRuntime(
        config=TranscriptionWorkerConfig.from_settings(settings),
        segment_queue=segment_queue,
        stop_event=stop,
        log_event=events.log,
        metrics=metrics,
        overload=overload,
        beam_controller=beam_controller,
        diarization=diarization,
        utterances=utterances,
        asr_backend_factory=dependencies.asr_backend_factory,
    )
    ingest = TapIngestRuntime(
        tap_queue=tap_queue,
        stop_event=stop,
        mode=settings.mode,
        segmenter=segmenter,
        log_event=events.log,
        emit_metrics=lambda force=False: worker.emit_metrics(force=bool(force)),
    )
    diarization_updates = _build_diarization_updates(
        settings=settings,
        segment_queue=diarization_queue,
        stop=stop,
        diarization=diarization,
        log_event=events.log,
    )
    return ASRRuntimeGraph(
        worker_runner=dependencies.worker_runner,
        stop=stop,
        logger=logger,
        events=events,
        overload=overload,
        diarization=diarization,
        utterances=utterances,
        metrics=metrics,
        segmenter_config=segmenter_config,
        segmenter=segmenter,
        worker=worker,
        diarization_updates=diarization_updates,
        ingest=ingest,
    )


def _diarization_sidecar_enabled(settings: ASRPipelineSettings) -> bool:
    return bool(settings.diarization_enabled and settings.diarization_sidecar_enabled)


def _build_diarization_queue(settings: ASRPipelineSettings) -> Optional["queue.Queue[Segment]"]:
    if not _diarization_sidecar_enabled(settings):
        return None
    return queue.Queue(maxsize=max(1, int(settings.diarization_queue_size)))


def _build_diarization_updates(
    *,
    settings: ASRPipelineSettings,
    segment_queue: Optional["queue.Queue[Segment]"],
    stop: StopSignalPort,
    diarization: DiarizationPort,
    log_event,
) -> Optional[DiarizationUpdateRuntime]:
    if segment_queue is None:
        return None
    return DiarizationUpdateRuntime(
        config=DiarizationUpdateConfig(
            enabled=_diarization_sidecar_enabled(settings),
            source_speaker_labels=dict(settings.source_speaker_labels or {}),
        ),
        segment_queue=segment_queue,
        stop_event=stop,
        diarization=diarization,
        log_event=log_event,
    )
