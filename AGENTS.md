# Agent Notes

This repository is a desktop Meeting Scribe app: Python backend, Electron main/preload, and React renderer. Keep context tight. Prefer reading the focused module for the current bug over scanning generated output, model files, coverage reports, or runtime logs.

Start with:
- `docs/PROJECT_MAP.md` for code ownership and runtime flows.
- `docs/TESTING.md` for the right verification command.
- `package.json` for npm scripts.
- `config.json` only when behavior depends on current local settings.

High-value commands:
- Search: `rg "<term>" backend/src frontend tests`
- Full tests: `npm test`
- Frontend tests: `npm run test:frontend`
- Backend tests: `python -m unittest discover -s tests -t . -v`
- Frontend coverage: `npm run test:coverage:frontend`
- Backend coverage: `npm run test:coverage:backend`
- Build renderer: `npm run build`

Working rules:
- The worktree is often dirty. Do not reset, checkout, or revert files unless explicitly asked.
- Do not touch `tools/batch_scribe.py` unless the user explicitly asks; it may be unrelated untracked work.
- Avoid reading `node_modules`, `models`, `build`, `dist`, `coverage`, `.local`, `.venv`, and large runtime logs unless the task specifically needs them.
- Use `rg` before opening files. Open the smallest likely file first.
- Use `apply_patch` for manual file edits.
- Keep new tests compact and split by behavior/module; do not dump unrelated cases into one large file.
- Put React tests under `frontend/renderer/tests/` mirroring `src/`, Electron tests under `frontend/electron/tests/`, and backend tests under `tests/backend/<area>/`.
- Prefer existing architecture and helper functions over new abstractions.
- Keep documentation and code edits ASCII unless the target file already relies on non-ASCII text.

Common routes:
- Transcript speaker bugs: `backend/src/interface/session_controller_parts/transcript_mixin.py`, `backend/src/transcription/domain/transcript_lines.py`, `backend/src/diarization/application/diarization_updates.py`
- ASR segmentation/quality timing: `backend/src/asr/infrastructure/segmentation.py`, `backend/src/asr/application/transcription_worker.py`, `backend/src/asr/domain/profiles.py`
- ASR runtime wiring: `backend/src/asr/application/runtime_graph.py`, `backend/src/infrastructure/asr_pipeline_factory.py`
- Electron IPC: `frontend/electron/main.cjs`, `frontend/electron/ipc-handlers.cjs`, `frontend/electron/preload-api.cjs`
- React state flow: `frontend/renderer/src/app/useMeetingScribeApp/`
- Settings mapping: `frontend/renderer/src/entities/settings/modelParts/`, `backend/src/interface/session_controller_parts/settings.py`
- Local LLM: `backend/src/infrastructure/local_llm.py`, `backend/src/infrastructure/local_llm_parts/`, `frontend/renderer/src/features/settings/assistant/`

Before final response:
- Run the narrow tests for touched code.
- Run `npm test` for shared behavior or cross-boundary changes.
- Run `git diff --check`.
- Mention any command that could not be run.
