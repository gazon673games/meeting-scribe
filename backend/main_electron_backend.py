from __future__ import annotations

import logging

# ruff: noqa: E402
# Runtime bootstrap must configure import paths before application imports.
import os
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent / 'src'
if _SRC_ROOT.exists():
    src_text = str(_SRC_ROOT)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


_logger = logging.getLogger(__name__)


def _configure_utf8_stdio() -> None:
    for stream_name in ('stdin', 'stdout', 'stderr'):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, 'reconfigure'):
            continue
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            _logger.debug(
                'Best-effort operation failed in _configure_utf8_stdio',
                exc_info=True,
            )


_configure_utf8_stdio()

from application.assistant_use_case import AssistantRequestUseCase
from application.local_paths import application_root, configure_project_local_io
from application.model_directories import model_directory
from application.session_tasks import OfflinePassUseCase, StopAsrSessionUseCase
from assistant.application.service import AssistantApplicationService
from infrastructure.asr_pipeline_factory import ASRPipelineFactory
from infrastructure.audio_runtime import DefaultAudioRuntimeFactory
from infrastructure.audio_source_factory import DefaultAudioSourceFactory
from infrastructure.background_tasks import model_download_tasks
from infrastructure.codex_cli import CodexCliRunner
from infrastructure.device_catalog import SoundDeviceCatalog
from infrastructure.llama_cpp_runner import LlamaCppRunner
from infrastructure.local_llm import (
    OllamaLocalLlmRunner,
    OpenAICompatibleLocalLlmRunner,
)
from infrastructure.local_model_runtime import shutdown as close_local_models
from infrastructure.offline_asr import FasterWhisperOfflineAsrRunner
from infrastructure.wav_recording import WavWriterFactory
from interface.assistant_controller import AssistantController
from interface.backend import ElectronBackend
from interface.jsonl_bridge import JsonLineBridge
from interface.session_controller import HeadlessSessionController
from settings.infrastructure.json_config_repository import JsonConfigRepository
from settings.infrastructure.runtime_config import ensure_runtime_config
from transcription.application.startup_service import (
    TranscriptionStartupService,
)
from transcription.infrastructure.file_transcript_context import (
    FileTranscriptContextReader,
)
from transcription.infrastructure.file_transcript_store import (
    FileTranscriptStore,
)


def runtime_root() -> Path:
    override = os.environ.get('MEETING_SCRIBE_RUNTIME_ROOT')
    if override:
        return Path(override).expanduser().resolve()
    return application_root()


def create_backend() -> ElectronBackend:
    project_root = runtime_root()
    configure_project_local_io(project_root)
    config_repository = JsonConfigRepository(project_root / 'config.json')
    ensure_runtime_config(project_root, config_repository)
    configure_project_local_io(
        project_root,
        models_dir=_models_dir_from_config(project_root, config_repository),
    )
    session_controller = HeadlessSessionController(
        project_root=project_root,
        audio_runtime_factory=DefaultAudioRuntimeFactory(),
        audio_source_factory=DefaultAudioSourceFactory(),
        wav_recorder_factory=WavWriterFactory(),
        asr_runtime_factory=ASRPipelineFactory(),
        transcription_startup_service=TranscriptionStartupService(),
        stop_asr_use_case=StopAsrSessionUseCase(),
        offline_pass_use_case=OfflinePassUseCase(
            FasterWhisperOfflineAsrRunner()
        ),
        transcript_store=FileTranscriptStore(project_root),
    )
    assistant_service = AssistantApplicationService(
        AssistantRequestUseCase(
            [
                CodexCliRunner(),
                OllamaLocalLlmRunner(),
                OpenAICompatibleLocalLlmRunner(),
                LlamaCppRunner(),
            ],
            FileTranscriptContextReader(),
        )
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


def _models_dir_from_config(
    project_root: Path, config_repository: JsonConfigRepository
) -> Path:
    try:
        config = config_repository.read()
    except (OSError, ValueError):
        _logger.warning('Cannot read model cache configuration', exc_info=True)
        config = {}
    return model_directory(
        project_root, config if isinstance(config, dict) else {}, 'asr'
    )


def main() -> None:
    if '--smoke-import' in sys.argv:
        return
    if '--repair-config' in sys.argv:
        root = runtime_root()
        configure_project_local_io(root)
        repo = JsonConfigRepository(root / 'config.json')
        ensure_runtime_config(root, repo)
        configure_project_local_io(
            root, models_dir=_models_dir_from_config(root, repo)
        )
        return

    backend = create_backend()
    bridge = JsonLineBridge(
        backend.handle, on_disconnect=backend.request_shutdown
    )
    backend.set_event_sink(bridge.emit_event)
    try:
        bridge.serve_forever()
    finally:
        _shutdown_backend(backend)


def _shutdown_backend(backend: ElectronBackend) -> None:
    failures: list[Exception] = []
    callbacks = [model_download_tasks.shutdown, close_local_models]
    if backend.session_controller is not None:
        callbacks.insert(0, backend.session_controller.shutdown)
    if backend.assistant_controller is not None:
        callbacks.insert(0, backend.assistant_controller.shutdown)
    for close in callbacks:
        try:
            close()
        except Exception as error:
            failures.append(error)
            _logger.exception('Backend shutdown operation failed')
    if failures and sys.exc_info()[0] is None:
        raise failures[0]


if __name__ == '__main__':
    main()
