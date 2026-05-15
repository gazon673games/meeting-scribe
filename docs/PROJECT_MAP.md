# Project Map

## Top Level

- `backend/` - Python backend runtime, application logic, ASR, diarization, persistence, and Electron bridge.
- `frontend/electron/` - Electron main process, preload API, IPC handlers, dev launcher, and app assets.
- `frontend/renderer/` - React UI and client-side state management.
- `tests/` - Python unit and integration-style backend tests.
- `tools/` - Release, coverage, and helper scripts.
- `requirements/` - Python dependency files.
- `config.json` - Local runtime configuration used by the app.

Avoid using these for broad context unless the task requires it: `.local/`, `models/`, `node_modules/`, `build/`, `dist/`, `coverage/`, `.venv/`.

## Backend

### Entry Points

- `backend/main_electron_backend.py` - backend process entry for Electron.
- `backend/src/interface/jsonl_bridge.py` - JSONL request/event bridge.
- `backend/src/interface/backend_impl.py` - backend object composition.
- `backend/src/interface/backend_parts/` - request handlers for state, sessions, models, downloads, and orchestration.

### Session And Transcript

- `backend/src/interface/session_controller.py` - headless session controller facade.
- `backend/src/interface/session_controller_parts/runtime_session_mixin.py` - start/stop/session snapshot behavior.
- `backend/src/interface/session_controller_parts/runtime_asr_mixin.py` - ASR runtime lifecycle and event wiring.
- `backend/src/interface/session_controller_parts/transcript_mixin.py` - ASR event dispatch into transcript lines.
- `backend/src/transcription/domain/transcript_lines.py` - transcript line IDs and speaker-update matching.
- `backend/src/transcription/infrastructure/file_transcript_store.py` - human logs and realtime SRT output.

### ASR

- `backend/src/infrastructure/asr_pipeline_factory.py` - converts session settings into ASR pipeline settings.
- `backend/src/asr/application/runtime_graph.py` - builds ingest, segmenter, worker, streaming path, and diarization sidecar.
- `backend/src/asr/infrastructure/segmentation.py` - VAD segmentation and overlap handling.
- `backend/src/asr/application/transcription_worker.py` - batch ASR worker, utterance aggregation, metrics.
- `backend/src/asr/infrastructure/streaming_segmenter.py` and `backend/src/asr/infrastructure/streaming_worker.py` - word-by-word streaming path.
- `backend/src/asr/domain/profiles.py` - Realtime, Quality, Ultra Fast profile defaults.

### Diarization And Speaker Identity

- `backend/src/diarization/application/diarization_updates.py` - sidecar worker that emits `transcript_speaker_update`.
- `backend/src/diarization/infrastructure/diarization_runtime.py` - runtime wrapper for online, sherpa, nemo, pyannote.
- `backend/src/diarization/infrastructure/diarizer.py` - online embedding clustering.
- `backend/src/identity/` - persistent speaker identity and matching.
- `backend/src/interface/session_controller_parts/artifacts_identity.py` - applies identity metadata to session outputs.

### Assistant And Local LLM

- `backend/src/application/assistant_use_case.py` - assistant request orchestration.
- `backend/src/interface/assistant_controller.py` - Electron-facing assistant controller.
- `backend/src/infrastructure/codex_cli.py` and `backend/src/infrastructure/codex_cli_parts/` - Codex CLI provider.
- `backend/src/infrastructure/local_llm.py` - Ollama and OpenAI-compatible local provider.
- `backend/src/infrastructure/local_llm_parts/runtime_server.py` - local llama-server autostart and port checks.

### Settings And Models

- `backend/src/interface/session_controller_parts/settings.py` - start-session params to `ASRSessionSettings`.
- `backend/src/application/codex_config.py` - assistant profile parsing.
- `backend/src/application/model_download.py`, `llm_model_download.py`, `diarization_model_download.py` - model catalogs/downloads.
- `backend/src/interface/backend_parts/model_state_mixin.py` - model/download state and selected-model checks.

## Frontend

### Electron

- `frontend/electron/main.cjs` - BrowserWindow setup and backend process.
- `frontend/electron/ipc-handlers.cjs` - main-process IPC registration.
- `frontend/electron/preload.cjs` - exposes preload API to renderer.
- `frontend/electron/preload-api.cjs` - testable preload API factory.
- `frontend/electron/run-electron.cjs` - launches Electron runtime.
- `frontend/electron/dev-runner.cjs` - dev runner that chooses a free renderer port.
- `frontend/electron/renderer-url.cjs` - shared renderer URL helpers.

### React Renderer

- `frontend/renderer/src/app/App.jsx` - top-level app composition.
- `frontend/renderer/src/app/useMeetingScribeApp.js` - main app hook.
- `frontend/renderer/src/app/useMeetingScribeApp/actions.js` - backend actions.
- `frontend/renderer/src/app/useMeetingScribeApp/backendEvents.js` - live event handling.
- `frontend/renderer/src/app/useMeetingScribeApp/transcriptState.js` - transcript state updates.
- `frontend/renderer/src/entities/settings/modelParts/` - config draft mapping, constants, and normalizers.
- `frontend/renderer/src/features/processing/` - profile, language, and processing controls.
- `frontend/renderer/src/features/settings/` - settings panels.
- `frontend/renderer/src/features/transcript/` - transcript UI.
- `frontend/renderer/src/features/assistant/` - assistant panel.

## Runtime Flows

### Start Session

1. React calls `start_session` with `draftToStartParams`.
2. Electron IPC forwards to backend.
3. `ElectronBackend` merges config and request params.
4. `HeadlessSessionController` builds `ASRSessionSettings`.
5. `ASRPipelineFactory` builds the ASR runtime graph.
6. Ingest receives audio packets and feeds the segmenter.
7. Segmenter emits `Segment` objects to ASR worker and optionally diarization sidecar.
8. ASR worker emits utterance events.
9. Transcript mixin appends lines and applies speaker updates.
10. Renderer receives `transcript_line` and `transcript_line_update`.

### Quality Mode Speaker Timing

Quality mode uses longer segments and overlap. With diarization sidecar enabled, `transcript_speaker_update` can arrive before ASR emits the matching transcript line. Speaker-update matching must avoid assigning a small overlap at the previous boundary to the wrong line; unmatched updates should stay pending until the correct line appears.

### Assistant Request

1. Renderer invokes `invoke_assistant`.
2. Backend assistant controller resolves profile and transcript context.
3. Assistant use case routes to Codex CLI, Ollama, OpenAI-compatible local, or local GGUF runner.
4. Result is emitted back as assistant state/event.

### Local LLM

- Ollama default: `http://127.0.0.1:11434`
- OpenAI-compatible default: `http://127.0.0.1:1234/v1`
- Defaults are editable through assistant profiles.
- For local llama-server autostart, port conflicts should produce actionable errors.

## Useful Test Anchors

- Transcript matching: `tests/test_transcript_lines.py`
- Session transcript/speaker updates: `tests/test_electron_interface_session.py`
- Diarization sidecar: `tests/test_diarization_updates.py`
- Diarization runtime: `tests/test_diarization_runtime.py`
- Electron IPC/preload: `frontend/electron/__tests__/`
- React app flow: `frontend/renderer/src/app/App.flow.test.jsx`
- Local LLM: `tests/test_local_llm.py`, `tests/test_local_llm_runtime_parts.py`
