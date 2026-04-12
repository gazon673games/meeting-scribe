from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from application.codex_use_case import CodexRequestUseCase
from infrastructure.asr_pipeline_factory import ASRPipelineFactory
from infrastructure.audio_source_factory import DefaultAudioSourceFactory
from infrastructure.codex_cli import CodexCliRunner
from infrastructure.device_catalog import SoundDeviceCatalog
from infrastructure.offline_asr import FasterWhisperOfflineAsrRunner
from infrastructure.wav_recording import WavWriterFactory
from ui.app import MainWindow


def create_main_window() -> MainWindow:
    codex_runner = CodexCliRunner()
    return MainWindow(
        asr_runtime_factory=ASRPipelineFactory(),
        audio_source_factory=DefaultAudioSourceFactory(),
        device_catalog=SoundDeviceCatalog(),
        wav_recorder_factory=WavWriterFactory(),
        codex_request_use_case=CodexRequestUseCase(codex_runner),
        offline_asr_runner=FasterWhisperOfflineAsrRunner(),
    )


def main() -> None:
    app = QApplication(sys.argv)
    window = create_main_window()
    window.show()
    sys.exit(app.exec())
