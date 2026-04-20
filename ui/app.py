# --- File: D:\work\own\voice2textTest\ui\app.py ---
from __future__ import annotations

import json
import queue
import sys
from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from audio.domain.formats import AudioFormat
from application.asr_profiles import (
    PROFILE_BALANCED as ASR_PROFILE_BALANCED,
    PROFILE_CUSTOM as ASR_PROFILE_CUSTOM,
    PROFILE_QUALITY as ASR_PROFILE_QUALITY,
    PROFILE_REALTIME as ASR_PROFILE_REALTIME,
)
from application.asr_session import ASRRuntimeFactory
from application.background_tasks import BackgroundTaskRunner
from application.audio_runtime import AudioRuntimeFactory
from application.audio_sources import AudioSourceFactory
from application.codex_use_case import CodexRequestUseCase
from application.device_catalog import DeviceCatalog
from application.recording import WavRecorderFactory
from application.session_tasks import OfflinePassUseCase, StopAsrSessionUseCase
from ui.asr_events_mixin import AsrEventsMixin
from ui.config_mixin import MainWindowConfigMixin
from ui.codex_integration import CodexIntegrationMixin
from ui.main_layout_mixin import MainWindowLayoutMixin
from ui.session_mixin import SessionMixin
from ui.source_controls_mixin import SourceControlsMixin, SourceRow
from ui.telemetry_mixin import TelemetryMixin
from ui.transcript_mixin import TranscriptMixin
from ui.window_helpers_mixin import WindowHelpersMixin

def ensure_runtime_config(project_root: Path, config_path: Path) -> None:
    if not getattr(sys, "frozen", False):
        return

    bundled_root = Path(getattr(sys, "_MEIPASS", project_root))
    bundled_config = bundled_root / "config.json"
    if not bundled_config.exists():
        return

    if not config_path.exists():
        try:
            config_path.write_text(bundled_config.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
        return

    try:
        current = json.loads(config_path.read_text(encoding="utf-8"))
        bundled = json.loads(bundled_config.read_text(encoding="utf-8"))
    except Exception:
        return

    if not isinstance(current, dict) or not isinstance(bundled, dict):
        return

    current_codex = current.get("codex", {})
    bundled_codex = bundled.get("codex", {})
    if not isinstance(current_codex, dict) or not isinstance(bundled_codex, dict):
        return

    has_profiles = bool(current_codex.get("profiles"))
    if "codex" in current and has_profiles:
        return

    current["codex"] = bundled_codex
    try:
        config_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


class MainWindow(
    SessionMixin,
    TelemetryMixin,
    AsrEventsMixin,
    TranscriptMixin,
    MainWindowConfigMixin,
    CodexIntegrationMixin,
    MainWindowLayoutMixin,
    SourceControlsMixin,
    WindowHelpersMixin,
    QWidget,
):
    background_event = Signal(object)

    PROFILE_REALTIME = ASR_PROFILE_REALTIME
    PROFILE_BALANCED = ASR_PROFILE_BALANCED
    PROFILE_QUALITY = ASR_PROFILE_QUALITY
    PROFILE_CUSTOM = ASR_PROFILE_CUSTOM

    def __init__(
        self,
        *,
        asr_runtime_factory: ASRRuntimeFactory,
        audio_runtime_factory: AudioRuntimeFactory,
        audio_source_factory: AudioSourceFactory,
        background_task_runner: BackgroundTaskRunner,
        device_catalog: DeviceCatalog,
        wav_recorder_factory: WavRecorderFactory,
        codex_request_use_case: CodexRequestUseCase,
        stop_asr_use_case: StopAsrSessionUseCase,
        offline_pass_use_case: OfflinePassUseCase,
    ):
        super().__init__()
        self.setWindowTitle("Meeting Scribe - Audio Mixer + ASR")
        self.resize(1180, 820)

        if getattr(sys, "frozen", False):
            self.project_root = Path(sys.executable).resolve().parent
        else:
            self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / "config.json"
        ensure_runtime_config(self.project_root, self.config_path)

        self.fmt = AudioFormat(sample_rate=48000, channels=2, dtype="float32", blocksize=1024)

        self.out_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)
        self.tap_q: "queue.Queue[dict]" = queue.Queue(maxsize=200)
        self.asr_ui_q: "queue.Queue[object]" = queue.Queue(maxsize=600)

        self.engine = audio_runtime_factory.create(format=self.fmt, output_queue=self.out_q, tap_queue=self.tap_q)
        self.rows: dict[str, SourceRow] = {}
        self.source_objs: Dict[str, Any] = {}

        self.asr_runtime_factory = asr_runtime_factory
        self.audio_runtime_factory = audio_runtime_factory
        self.audio_source_factory = audio_source_factory
        self.background_task_runner = background_task_runner
        self.device_catalog = device_catalog
        self.wav_recorder_factory = wav_recorder_factory
        self.codex_request_use_case = codex_request_use_case
        self.stop_asr_use_case = stop_asr_use_case
        self.offline_pass_use_case = offline_pass_use_case

        self.writer = self.wav_recorder_factory.create(self.out_q)
        self.writer.start()

        self.output_name = "capture_mix.wav"

        self._init_session_state()
        self._init_transcript_state()
        self._init_codex_state()
        self._init_telemetry_state()
        self.background_event.connect(self._handle_background_event)

        self._build_main_layout()
