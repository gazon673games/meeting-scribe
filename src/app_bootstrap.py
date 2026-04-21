from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent
if _SRC_ROOT.exists():
    src_text = str(_SRC_ROOT)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)

from PySide6.QtWidgets import QApplication

from application.local_paths import application_root
from application.codex_use_case import CodexRequestUseCase
from application.session_tasks import OfflinePassUseCase, StopAsrSessionUseCase
from assistant.application.service import AssistantApplicationService
from infrastructure.asr_pipeline_factory import ASRPipelineFactory
from infrastructure.audio_runtime import DefaultAudioRuntimeFactory
from infrastructure.audio_source_factory import DefaultAudioSourceFactory
from infrastructure.background_tasks import ThreadBackgroundTaskRunner
from infrastructure.codex_cli import CodexCliRunner
from infrastructure.device_catalog import SoundDeviceCatalog
from infrastructure.offline_asr import FasterWhisperOfflineAsrRunner
from infrastructure.wav_recording import WavWriterFactory
from settings.infrastructure.json_config_repository import JsonConfigRepository
from settings.infrastructure.runtime_config import ensure_runtime_config
from transcription.application.startup_service import TranscriptionStartupService
from transcription.infrastructure.file_transcript_context import FileTranscriptContextReader
from transcription.infrastructure.file_transcript_store import FileTranscriptStore
from ui.app import MainWindow


def create_main_window() -> MainWindow:
    project_root = application_root()
    config_repository = JsonConfigRepository(project_root / "config.json")
    ensure_runtime_config(project_root, config_repository)

    codex_runner = CodexCliRunner()
    codex_use_case = CodexRequestUseCase(codex_runner, FileTranscriptContextReader())
    offline_asr_runner = FasterWhisperOfflineAsrRunner()
    background_tasks = ThreadBackgroundTaskRunner()
    return MainWindow(
        asr_runtime_factory=ASRPipelineFactory(),
        audio_runtime_factory=DefaultAudioRuntimeFactory(),
        audio_source_factory=DefaultAudioSourceFactory(),
        background_task_runner=background_tasks,
        device_catalog=SoundDeviceCatalog(),
        wav_recorder_factory=WavWriterFactory(),
        assistant_service=AssistantApplicationService(codex_use_case),
        transcription_startup_service=TranscriptionStartupService(),
        config_repository=config_repository,
        transcript_store=FileTranscriptStore(project_root),
        stop_asr_use_case=StopAsrSessionUseCase(),
        offline_pass_use_case=OfflinePassUseCase(offline_asr_runner),
    )


def main() -> None:
    app = QApplication(sys.argv)
    window = create_main_window()
    window.show()
    sys.exit(app.exec())
