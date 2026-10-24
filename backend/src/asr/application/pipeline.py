from __future__ import annotations

import logging
import queue
import time
from pathlib import Path

from asr.application.pipeline_config import (
    ASRPipelineDependencies,
    ASRPipelineSettings,
)
from asr.application.runtime_graph import ASRRuntimeGraph, build_runtime_graph

_logger = logging.getLogger(__name__)


class ASRPipeline:
    """
    Lifecycle facade for realtime ASR.

    Configuration lives in ASRPipelineSettings, and collaborator wiring lives in
    runtime_graph. This class intentionally keeps only the public runtime shape:
    construct, start, stop.
    """

    def __init__(
        self,
        *,
        tap_queue: queue.Queue[dict],
        project_root: Path,
        settings: ASRPipelineSettings,
        dependencies: ASRPipelineDependencies,
        ui_queue: queue.Queue[dict] | None = None,
        event_queue: queue.Queue[object] | None = None,
    ) -> None:
        self.tap_q = tap_queue
        self.project_root = project_root
        self.settings = settings
        self.session_id = f'sess_{int(time.time())}'
        self.graph: ASRRuntimeGraph = build_runtime_graph(
            settings=settings,
            dependencies=dependencies,
            tap_queue=tap_queue,
            project_root=project_root,
            session_id=self.session_id,
            ui_queue=ui_queue,
            event_queue=event_queue,
        )

        self.language = settings.language
        self.mode = settings.mode
        self.source_names = settings.source_names
        self.asr_model_name = settings.asr_model_name
        self.device = settings.device
        self.compute_type = settings.compute_type
        self.beam_size = int(settings.beam_size)
        self.asr_language = settings.asr_language
        self.asr_initial_prompt = settings.asr_initial_prompt
        self.logger = self.graph.logger

    def start(self) -> None:
        self.graph.start(settings=self.settings, session_id=self.session_id)

    def stop(self) -> None:
        self.graph.stop_runtime()

    def identity_snapshot(self) -> dict:
        try:
            return dict(self.graph.diarization.identity_snapshot() or {})
        except (OSError, RuntimeError):
            _logger.warning(
                'Could not export speaker identities', exc_info=True
            )
            return {}
