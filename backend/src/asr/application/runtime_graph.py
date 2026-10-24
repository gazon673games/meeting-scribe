from __future__ import annotations

import gc
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from asr.application.events import ASREventPublisher
from asr.application.ingest import TapIngestRuntime
from asr.application.metrics import ASRMetrics
from asr.application.overload import OverloadController
from asr.application.pipeline_config import (
    ASRPipelineDependencies,
    ASRPipelineSettings,
    build_diarization_config,
    build_segmenter_config,
    build_streaming_segmenter_config,
)
from asr.application.policies import AdaptiveBeam
from asr.application.ports import (
    AsrLoggerPort,
    RealtimeWorkerRunnerPort,
    StopSignalPort,
    WorkerHandlePort,
)
from asr.application.segmentation import AudioSegmenterPort, SegmenterConfig
from asr.application.streaming_worker_config import StreamingWorkerConfig
from asr.application.transcription_worker import TranscriptionWorkerRuntime
from asr.application.utterances import UtteranceAggregator
from asr.application.worker_config import TranscriptionWorkerConfig
from asr.domain.segments import Segment
from asr.infrastructure.streaming_queue import CoalescingStreamingQueue
from asr.infrastructure.streaming_segmenter import StreamingAudioSegmenter
from asr.infrastructure.streaming_worker import StreamingWhisperWorker
from diarization.application.diarization import DiarizationPort
from diarization.application.diarization_updates import (
    DiarizationUpdateConfig,
    DiarizationUpdateRuntime,
)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ASRRuntimePath:
    segmenter: AudioSegmenterPort
    worker: TranscriptionWorkerRuntime | None = None
    streaming_worker: StreamingWhisperWorker | None = None
    diarization_updates: DiarizationUpdateRuntime | None = None


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
    worker: TranscriptionWorkerRuntime | None
    diarization_updates: DiarizationUpdateRuntime | None
    ingest: TapIngestRuntime
    ingest_worker: WorkerHandlePort | None = None
    transcription_worker: WorkerHandlePort | None = None
    diarization_worker: WorkerHandlePort | None = None
    streaming_worker_runtime: StreamingWhisperWorker | None = None
    streaming_worker_handle: WorkerHandlePort | None = None
    _running: bool = field(default=False, init=False, repr=False)
    _lifecycle_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )

    def start(self, *, settings: ASRPipelineSettings, session_id: str) -> None:
        with self._lifecycle_lock:
            if self._running:
                return
            self._running = True
            try:
                self._start_workers(settings=settings, session_id=session_id)
            except BaseException:
                try:
                    self.stop_runtime()
                except Exception:
                    _logger.exception('Cannot finish ASR startup rollback')
                raise

    def _start_workers(
        self, *, settings: ASRPipelineSettings, session_id: str
    ) -> None:
        self.stop.clear()
        self.overload.reset()
        self.metrics.reset()
        self.segmenter.reset_runtime()
        if self.worker is not None:
            self.worker.reset_runtime()
        if self.streaming_worker_runtime is not None:
            self.streaming_worker_runtime.reset_runtime()
        if self.diarization_updates is not None:
            self.diarization_updates.reset_runtime()

        self.events.log(
            self._build_started_event(settings=settings, session_id=session_id)
        )
        self.ingest_worker = self.worker_runner.start_worker(
            name='asr-ingest', target=self.ingest.run_safe
        )
        if self.worker is not None:
            self.transcription_worker = self.worker_runner.start_worker(
                name='asr-worker',
                target=self.worker.run_safe,
            )
        if self.streaming_worker_runtime is not None:
            self.streaming_worker_handle = self.worker_runner.start_worker(
                name='asr-streaming',
                target=self.streaming_worker_runtime.run_safe,
            )
        if self.diarization_updates is not None:
            self.diarization_worker = self.worker_runner.start_worker(
                name='asr-diarization',
                target=self.diarization_updates.run_safe,
            )
        self._wait_for_asr_initialization(float(settings.startup_timeout_s))

    def stop_runtime(self) -> None:
        with self._lifecycle_lock:
            if not self._running:
                return
            # Workers do not acquire this lock. Serialize restart with all joins.
            self.stop.set()
            failures: list[Exception] = []
            for handle in (
                self.ingest_worker,
                self.transcription_worker,
                self.streaming_worker_handle,
                self.diarization_worker,
            ):
                if handle is None:
                    continue
                try:
                    handle.join()
                except Exception as error:
                    failures.append(error)
                    _logger.exception('Cannot join ASR worker')
            if failures:
                # Retain handles and prevent restart until cleanup can be retried.
                raise failures[0]
            self.ingest_worker = None
            self.transcription_worker = None
            self.streaming_worker_handle = None
            self.diarization_worker = None
            failure: BaseException | None = None
            try:
                if self.worker is not None:
                    try:
                        self.worker.flush_utterances(force=True)
                    except Exception:
                        _logger.exception('Cannot flush final ASR utterances')
                    self.worker.emit_metrics(force=True)
                self.events.log({'type': 'asr_stopped', 'ts': time.time()})
            except BaseException as error:
                failure = error
            try:
                self.logger.close()
            except BaseException:
                if failure is not None:
                    _logger.exception(
                        'Cannot close ASR logger after shutdown failure'
                    )
                    raise failure
                raise
            self._running = False
            gc.collect()
            if failure is not None:
                raise failure

    def _wait_for_asr_initialization(self, timeout_s: float) -> None:
        runtime = (
            self.worker
            if self.worker is not None
            else self.streaming_worker_runtime
        )
        if runtime is None:
            return
        runtime.wait_initialized(timeout_s)

    def segmentation_params(
        self, settings: ASRPipelineSettings
    ) -> tuple[float, float, float]:
        return self.overload.segmentation_params(
            endpoint_silence_ms=settings.endpoint_silence_ms,
            max_segment_s=settings.max_segment_s,
            overlap_ms=settings.overlap_ms,
        )

    def _build_started_event(
        self, *, settings: ASRPipelineSettings, session_id: str
    ) -> dict:
        return {
            'type': 'asr_started',
            'session_id': session_id,
            'language': settings.language,
            'mode': settings.mode,
            'model': settings.asr_model_name,
            'device': settings.device,
            'compute_type': settings.compute_type,
            'cpu_threads': int(settings.cpu_threads),
            'num_workers': int(settings.num_workers),
            'beam_size': int(settings.beam_size),
            'endpoint_silence_ms': settings.endpoint_silence_ms,
            'max_segment_s': settings.max_segment_s,
            'overlap_ms': settings.overlap_ms,
            'overload_strategy': self.overload.strategy,
            'vad': self.segmenter_config.to_event_dict(),
            'overload': self.overload.to_event_dict(),
            'diarization_enabled': self.diarization.enabled,
            'diar_backend': self.diarization.backend,
            'agc_enabled': self.segmenter_config.agc_enabled,
            'text_dedup_enabled': settings.text_dedup_enabled,
            'adaptive_beam_enabled': settings.adaptive_beam_enabled,
            **self.utterances.to_event_dict(),
            'log_rotation': {
                'max_bytes': self.logger.max_bytes,
                'backup_count': self.logger.backup_count,
            },
            'log_speaker_labels': settings.log_speaker_labels,
            'asr_language': settings.asr_language,
            'asr_initial_prompt': bool(settings.asr_initial_prompt),
            'asr_hotwords': bool(settings.asr_hotwords),
            'streaming_enabled': settings.streaming_enabled,
            'ts': time.time(),
        }


def build_runtime_graph(
    *,
    settings: ASRPipelineSettings,
    dependencies: ASRPipelineDependencies,
    tap_queue: queue.Queue[dict],
    project_root: Path,
    session_id: str,
    ui_queue: queue.Queue[dict] | None = None,
    event_queue: queue.Queue[object] | None = None,
) -> ASRRuntimeGraph:
    stop = dependencies.worker_runner.create_stop_signal()

    logger = dependencies.logger_factory(
        root=project_root,
        session_id=session_id,
        language=settings.language,
        max_bytes=int(settings.log_max_bytes),
        backup_count=int(settings.log_backup_count),
    )
    try:
        return _assemble_runtime_graph(
            settings=settings,
            dependencies=dependencies,
            stop=stop,
            logger=logger,
            tap_queue=tap_queue,
            project_root=project_root,
            ui_queue=ui_queue,
            event_queue=event_queue,
        )
    except BaseException:
        try:
            logger.close()
        except Exception:
            _logger.exception(
                'Cannot close logger after ASR graph construction failure'
            )
        raise


def _assemble_runtime_graph(
    *,
    settings: ASRPipelineSettings,
    dependencies: ASRPipelineDependencies,
    stop: StopSignalPort,
    logger: AsrLoggerPort,
    tap_queue: queue.Queue[dict],
    project_root: Path,
    ui_queue: queue.Queue[dict] | None,
    event_queue: queue.Queue[object] | None,
) -> ASRRuntimeGraph:
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

    if settings.streaming_enabled:
        runtime_path = _build_streaming_path(
            settings=settings,
            dependencies=dependencies,
            stop=stop,
            events=events,
            metrics=metrics,
        )
    else:
        runtime_path = _build_batch_path(
            settings=settings,
            dependencies=dependencies,
            stop=stop,
            events=events,
            metrics=metrics,
            overload=overload,
            beam_controller=beam_controller,
            diarization=diarization,
            utterances=utterances,
        )

    ingest = TapIngestRuntime(
        tap_queue=tap_queue,
        stop_event=stop,
        mode=settings.mode,
        segmenter=runtime_path.segmenter,
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
        segmenter=runtime_path.segmenter,
        worker=runtime_path.worker,
        diarization_updates=runtime_path.diarization_updates,
        ingest=ingest,
        streaming_worker_runtime=runtime_path.streaming_worker,
    )


def _build_streaming_path(
    *,
    settings: ASRPipelineSettings,
    dependencies: ASRPipelineDependencies,
    stop: StopSignalPort,
    events: ASREventPublisher,
    metrics: ASRMetrics,
) -> ASRRuntimePath:
    chunk_queue = CoalescingStreamingQueue(maxsize=50)
    segmenter = StreamingAudioSegmenter(
        config=build_streaming_segmenter_config(settings),
        chunk_queue=chunk_queue,
        log_event=events.log,
    )
    streaming_worker = StreamingWhisperWorker(
        config=StreamingWorkerConfig.from_settings(settings),
        chunk_queue=chunk_queue,
        stop_event=stop,
        log_event=events.log,
        asr_backend_factory=dependencies.asr_backend_factory,
        metrics=metrics,
    )
    return ASRRuntimePath(
        segmenter=segmenter, streaming_worker=streaming_worker
    )


def _build_batch_path(
    *,
    settings: ASRPipelineSettings,
    dependencies: ASRPipelineDependencies,
    stop: StopSignalPort,
    events: ASREventPublisher,
    metrics: ASRMetrics,
    overload: OverloadController,
    beam_controller: AdaptiveBeam,
    diarization: DiarizationPort,
    utterances: UtteranceAggregator,
) -> ASRRuntimePath:
    segment_queue: queue.Queue[Segment] = queue.Queue(maxsize=50)
    diarization_queue = _build_diarization_queue(settings)

    def segmentation_params() -> tuple[float, float, float]:
        return overload.segmentation_params(
            endpoint_silence_ms=settings.endpoint_silence_ms,
            max_segment_s=settings.max_segment_s,
            overlap_ms=settings.overlap_ms,
        )

    segmenter = dependencies.segmenter_factory(
        config=build_segmenter_config(settings),
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
    diarization_updates = _build_diarization_updates(
        settings=settings,
        segment_queue=diarization_queue,
        stop=stop,
        diarization=diarization,
        log_event=events.log,
    )
    return ASRRuntimePath(
        segmenter=segmenter,
        worker=worker,
        diarization_updates=diarization_updates,
    )


def _diarization_sidecar_enabled(settings: ASRPipelineSettings) -> bool:
    return bool(
        settings.diarization_enabled and settings.diarization_sidecar_enabled
    )


def _build_diarization_queue(
    settings: ASRPipelineSettings,
) -> queue.Queue[Segment] | None:
    if not _diarization_sidecar_enabled(settings):
        return None
    return queue.Queue(maxsize=max(1, int(settings.diarization_queue_size)))


def _build_diarization_updates(
    *,
    settings: ASRPipelineSettings,
    segment_queue: queue.Queue[Segment] | None,
    stop: StopSignalPort,
    diarization: DiarizationPort,
    log_event: Callable[[dict], None],
) -> DiarizationUpdateRuntime | None:
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
