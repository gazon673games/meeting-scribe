from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from application.asr_session import ASRRuntime, ASRRuntimeFactory, ASRSessionSettings
from application.audio_runtime import AudioRuntimeFactory, AudioRuntimePort
from application.audio_sources import AudioSourceFactory
from application.events import TranscriptSpeakerUpdateEvent
from application.recording import WavRecorder, WavRecorderFactory
from application.session_tasks import OfflinePassUseCase, StopAsrSessionUseCase
from audio.domain.formats import AudioFormat
from interface.session_controller_parts import (
    EventSink,
    HeadlessSource,
    RuntimeLifecycleMixin,
    SourceControlMixin,
    TranscriptPipelineMixin,
)
from interface.session_controller_parts.settings import asr_settings_from_params as _asr_settings_from_params
from session.domain.aggregate import SessionAggregate
from transcription.application.startup_service import TranscriptionStartupService
from transcription.application.transcript_store import TranscriptStore
from transcription.domain.aggregate import TranscriptionJobAggregate


class HeadlessSessionController(
    RuntimeLifecycleMixin,
    TranscriptPipelineMixin,
    SourceControlMixin,
):
    def __init__(
        self,
        *,
        project_root: Path,
        audio_runtime_factory: AudioRuntimeFactory,
        audio_source_factory: AudioSourceFactory,
        wav_recorder_factory: WavRecorderFactory,
        asr_runtime_factory: Optional[ASRRuntimeFactory] = None,
        transcription_startup_service: Optional[TranscriptionStartupService] = None,
        stop_asr_use_case: Optional[StopAsrSessionUseCase] = None,
        offline_pass_use_case: Optional[OfflinePassUseCase] = None,
        transcript_store: Optional[TranscriptStore] = None,
        event_sink: Optional[EventSink] = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.audio_runtime_factory = audio_runtime_factory
        self.audio_source_factory = audio_source_factory
        self.wav_recorder_factory = wav_recorder_factory
        self.asr_runtime_factory = asr_runtime_factory
        self.transcription_startup_service = transcription_startup_service
        self.stop_asr_use_case = stop_asr_use_case or StopAsrSessionUseCase()
        self.offline_pass_use_case = offline_pass_use_case
        self.transcript_store = transcript_store
        self.event_sink = event_sink

        self.format = AudioFormat(sample_rate=48000, channels=2, dtype="float32", blocksize=1024)
        self.output_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)
        self.tap_queue: "queue.Queue[dict]" = queue.Queue(maxsize=200)
        self.asr_event_queue: "queue.Queue[object]" = queue.Queue(maxsize=600)
        self.engine: AudioRuntimePort = self.audio_runtime_factory.create(
            format=self.format,
            output_queue=self.output_queue,
            tap_queue=None,
        )
        self.writer: WavRecorder = self.wav_recorder_factory.create(self.output_queue)
        self.writer.start()

        self._lock = threading.RLock()
        self._session_state = SessionAggregate()
        self._transcription_state = TranscriptionJobAggregate()
        self._sources: dict[str, HeadlessSource] = {}
        self._source_objs: dict[str, Any] = {}
        self._status = "idle"
        self._last_error = ""
        self._last_warning = ""
        self._asr_requested = False
        self._asr_running = False
        self._asr_runtime: Optional[ASRRuntime] = None
        self._asr_event_stop = threading.Event()
        self._asr_event_thread: Optional[threading.Thread] = None
        self._transcript_lines: list[Dict[str, Any]] = []
        self._pending_speaker_updates: list[TranscriptSpeakerUpdateEvent] = []
        self._realtime_transcript_enabled = False
        self._human_log_path: Optional[Path] = None
        self._realtime_transcript_path: Optional[Path] = None
        self._asr_metrics: Dict[str, Any] = {
            "segDroppedTotal": 0,
            "segSkippedTotal": 0,
            "avgLatencyS": 0.0,
            "p95LatencyS": 0.0,
            "lagS": 0.0,
        }
        self._offline_pass_thread: Optional[threading.Thread] = None
        self._offline_pass_running = False
        self._offline_pass_result: Dict[str, Any] = {}
        self._model_download_info: Dict[str, Any] = {}
        self._active_asr_settings: Optional[ASRSessionSettings] = None
        self._active_asr_session_id: str = ""
        self._closing = False


__all__ = ["HeadlessSessionController", "HeadlessSource", "_asr_settings_from_params"]

