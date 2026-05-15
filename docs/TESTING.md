# Testing

This project has two test stacks:
- Frontend/Electron tests use Vitest.
- Backend tests use Python `unittest`.

Run commands from the repository root.

## Quick Commands

```powershell
npm run test:frontend
```

```powershell
python -m unittest discover -s tests -t . -v
```

```powershell
npm test
```

`npm test` runs the frontend tests first, then the backend test suite.

## Coverage

Frontend coverage:

```powershell
npm run test:coverage:frontend
```

Output:
- terminal summary
- `coverage/frontend/`

Backend coverage:

```powershell
npm run test:coverage:backend
```

Output:
- terminal summary
- `coverage/backend/`

Combined coverage command:

```powershell
npm run test:coverage
```

Backend coverage requires `coverage.py` in the active Python environment. The project includes `requirements/requirements-dev.txt` for dev/test dependencies.

## Recommended Verification By Change Type

### Backend Domain Or Application Logic

Run the focused test file first:

```powershell
python -m unittest tests.test_transcript_lines -v
```

Then run backend tests:

```powershell
python -m unittest discover -s tests -t . -v
```

### ASR, Diarization, Transcript, Session Controller

Useful focused suites:

```powershell
python -m unittest tests.test_transcript_lines tests.test_electron_interface_session -v
```

```powershell
python -m unittest tests.test_diarization_updates tests.test_diarization_runtime -v
```

Then run:

```powershell
npm test
```

### Electron Main Or Preload IPC

Run:

```powershell
npm run test:frontend
```

Relevant tests live in:
- `frontend/electron/__tests__/ipc-handlers.test.js`
- `frontend/electron/__tests__/preload-api.test.js`
- `frontend/electron/__tests__/renderer-dev-url.test.js`

### React UI Behavior

Run:

```powershell
npm run test:frontend
```

Relevant tests live near the feature or app flow:
- `frontend/renderer/src/app/App.flow.test.jsx`
- `frontend/renderer/src/features/assistant/AssistantColumn.test.jsx`
- `frontend/renderer/src/features/transcript/TranscriptList.test.jsx`

### Config, Settings, Or Start Params

Run:

```powershell
python -m unittest tests.test_runtime_config tests.test_commands tests.test_electron_interface_session -v
```

For frontend settings mapping:

```powershell
npm run test:frontend
```

### Local LLM

Run:

```powershell
python -m unittest tests.test_local_llm tests.test_local_llm_runtime_parts -v
```

Also run frontend tests if assistant profile UI changed:

```powershell
npm run test:frontend
```

## Build Checks

Renderer build:

```powershell
npm run build
```

Whitespace check:

```powershell
git diff --check
```

`git diff --check` may print CRLF warnings on Windows. Treat actual whitespace errors as blockers.

## Notes For Agents

- Prefer focused tests while iterating, then run `npm test` before finalizing broad changes.
- Do not use `--runInBand`; that is a Jest flag and Vitest rejects it.
- Avoid generating coverage unless the user asks for coverage or the task is coverage-related.
- Do not inspect `coverage/`, `node_modules/`, `.local/`, `models/`, or build artifacts unless the task specifically needs them.
