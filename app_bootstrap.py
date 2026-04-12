from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from application.codex_use_case import CodexRequestUseCase
from application.session_tasks import OfflinePassUseCase, StopAsrSessionUseCase
from infrastructure.asr_pipeline_factory import ASRPipelineFactory
from infrastructure.audio_runtime import DefaultAudioRuntimeFactory
from infrastructure.audio_source_factory import DefaultAudioSourceFactory
from infrastructure.background_tasks import ThreadBackgroundTaskRunner
from infrastructure.codex_cli import CodexCliRunner
from infrastructure.device_catalog import SoundDeviceCatalog
from infrastructure.offline_asr import FasterWhisperOfflineAsrRunner
from infrastructure.wav_recording import WavWriterFactory
from ui.app import MainWindow


def create_main_window() -> MainWindow:
    codex_runner = CodexCliRunner()
    offline_asr_runner = FasterWhisperOfflineAsrRunner()
    background_tasks = ThreadBackgroundTaskRunner()
    return MainWindow(
        asr_runtime_factory=ASRPipelineFactory(),
        audio_runtime_factory=DefaultAudioRuntimeFactory(),
        audio_source_factory=DefaultAudioSourceFactory(),
        background_task_runner=background_tasks,
        device_catalog=SoundDeviceCatalog(),
        wav_recorder_factory=WavWriterFactory(),
        codex_request_use_case=CodexRequestUseCase(codex_runner),
        stop_asr_use_case=StopAsrSessionUseCase(),
        offline_pass_use_case=OfflinePassUseCase(offline_asr_runner),
    )


def main() -> None:
    app = QApplication(sys.argv)
    window = create_main_window()
    window.show()
    sys.exit(app.exec())
