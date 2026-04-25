from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent / "src"
if _SRC_ROOT.exists():
    src_text = str(_SRC_ROOT)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)

from application.codex_use_case import CodexRequestUseCase
from application.local_paths import application_root, configure_project_local_io
from application.session_tasks import OfflinePassUseCase, StopAsrSessionUseCase
from assistant.application.service import AssistantApplicationService
from infrastructure.asr_pipeline_factory import ASRPipelineFactory
from infrastructure.audio_runtime import DefaultAudioRuntimeFactory
from infrastructure.audio_source_factory import DefaultAudioSourceFactory
from infrastructure.codex_cli import CodexCliRunner
from infrastructure.device_catalog import SoundDeviceCatalog
from infrastructure.offline_asr import FasterWhisperOfflineAsrRunner
from infrastructure.wav_recording import WavWriterFactory
from interface.assistant_controller import AssistantController
from interface.backend import ElectronBackend
from interface.jsonl_bridge import JsonLineBridge
from interface.session_controller import HeadlessSessionController
from settings.infrastructure.json_config_repository import JsonConfigRepository
from settings.infrastructure.runtime_config import ensure_runtime_config
from transcription.application.startup_service import TranscriptionStartupService
from transcription.infrastructure.file_transcript_context import FileTranscriptContextReader


def create_backend() -> ElectronBackend:
    project_root = application_root()
    configure_project_local_io(project_root)
    config_repository = JsonConfigRepository(project_root / "config.json")
    ensure_runtime_config(project_root, config_repository)
    session_controller = HeadlessSessionController(
        project_root=project_root,
        audio_runtime_factory=DefaultAudioRuntimeFactory(),
        audio_source_factory=DefaultAudioSourceFactory(),
        wav_recorder_factory=WavWriterFactory(),
        asr_runtime_factory=ASRPipelineFactory(),
        transcription_startup_service=TranscriptionStartupService(),
        stop_asr_use_case=StopAsrSessionUseCase(),
        offline_pass_use_case=OfflinePassUseCase(FasterWhisperOfflineAsrRunner()),
    )
    assistant_service = AssistantApplicationService(
        CodexRequestUseCase(CodexCliRunner(), FileTranscriptContextReader())
    )
    assistant_controller = AssistantController(
        project_root=project_root,
        config_repository=config_repository,
        assistant_service=assistant_service,
        session_controller=session_controller,
    )
    return ElectronBackend(
        project_root=project_root,
        config_repository=config_repository,
        device_catalog=SoundDeviceCatalog(),
        session_controller=session_controller,
        assistant_controller=assistant_controller,
    )


def main() -> None:
    backend = create_backend()
    bridge = JsonLineBridge(backend.handle)
    backend.set_event_sink(bridge.emit_event)
    try:
        bridge.serve_forever()
    finally:
        if backend.session_controller is not None:
            backend.session_controller.shutdown()


if __name__ == "__main__":
    main()
