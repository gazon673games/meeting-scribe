# Testing

This project has two test stacks:
- Frontend/Electron tests use Vitest.
- Backend tests use Python `unittest`.

Run commands from the repository root.

## Test Layout

- React renderer tests live in `frontend/renderer/tests/` and mirror `frontend/renderer/src/`.
- Electron main/preload tests live in `frontend/electron/tests/`.
- Backend Python tests live in `tests/backend/<area>/`; shared test helpers live in `tests/helpers/`.
- Keep new tests out of production source folders and avoid adding flat root-level test files.

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
python -m unittest tests.backend.domain.test_transcript_lines -v
```

Then run backend tests:

```powershell
python -m unittest discover -s tests -t . -v
```

### ASR, Diarization, Transcript, Session Controller

Useful focused suites:

```powershell
python -m unittest tests.backend.domain.test_transcript_lines tests.backend.interface.test_electron_interface_session -v
```

```powershell
python -m unittest tests.backend.diarization.test_diarization_updates tests.backend.diarization.test_diarization_runtime -v
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
- `frontend/electron/tests/ipc-handlers.test.js`
- `frontend/electron/tests/preload-api.test.js`
- `frontend/electron/tests/renderer-dev-url.test.js`

### React UI Behavior

Run:

```powershell
npm run test:frontend
```

Relevant tests live near the feature or app flow:
- `frontend/renderer/tests/app/App.flow.test.jsx`
- `frontend/renderer/tests/features/assistant/AssistantColumn.test.jsx`
- `frontend/renderer/tests/features/transcript/TranscriptList.test.jsx`

### Config, Settings, Or Start Params

Run:

```powershell
python -m unittest tests.backend.infrastructure.test_runtime_config tests.backend.application.test_commands tests.backend.interface.test_electron_interface_session -v
```

For frontend settings mapping:

```powershell
npm run test:frontend
```

### Local LLM

Run:

```powershell
python -m unittest tests.backend.assistant.test_local_llm tests.backend.assistant.test_local_llm_runtime_parts -v
```

Also run frontend tests if assistant profile UI changed:

```powershell
npm run test:frontend
```

## Build Checks

Metarhia style checks for all JavaScript/JSX, Electron and renderer tests,
CSS, Python source and tests, and the PyInstaller specification:

```powershell
npm run lint
npm run fix
```

`lint` runs the JavaScript checks and Python checks. Install the Python development
requirements before running it:

```powershell
python -m pip install -r requirements/requirements-dev.txt
```

Run either language independently:

```powershell
npm run lint:frontend
npm run lint:python
npm run fix:frontend
npm run fix:python
```

`fix` includes test files. Test edits were explicitly authorized for this audit;
future test edits still follow `AGENTS.md`. Ruff covers `backend/`, `tools/`,
`tests/`, and the PyInstaller specification, including
`tools/batch_scribe.py`. Its rules and formatting implement the Python
preferences in `PROJECT_MAP.md`.
The ESLint configuration retains external snake_case protocol properties and
lets Prettier control expression parentheses.
JSX component references are tracked without suppressing unused imports.
Context-free JavaScript functions use arrows. ESLint rejects free function
declarations and ordinary free function expressions; generators and receiver
methods retain their required forms. Strict use-before-definition checks stay
enabled, so helpers must precede consumers without reordering initialization.
React effects may return a cleanup function or an explicit `undefined`.
Long import paths, test titles, URLs, and regular expressions are exempt from
the 80-column JavaScript limit. Other expressions keep the same limit.
The matching `.gitattributes` entries keep these modules in LF format on Windows.

Check frontend and Python dependency boundaries:

```powershell
npm run quality:deps
```

Run all style, dependency, and complexity checks:

```powershell
npm run quality
```

The complexity check runs both `tools/quality/complexity_gate.py` and
`frontend/test_complexity.js`. Both reject functions with complexity 21 or
higher. The existing complexity budgets are 2 percent for Python and 5 percent
for JavaScript. Parse failures fail the checks. Run them separately with
`npm run quality:complexity`. The Quality workflow runs the same command group
for pushes and pull requests.

Renderer build:

```powershell
npm run build
```

Whitespace check:

```powershell
git diff --check
```

`git diff --check` may print CRLF warnings on Windows. Treat actual whitespace errors as blockers.

On Windows, use `npm.cmd` if PowerShell blocks `npm.ps1`. If the workspace uses a
drive alias and Vite reports a missing `/@fs/` setup file on another drive, run
the same commands from the physical project path. It can be checked with:

```powershell
node -p "require('node:fs').realpathSync('.')"
```

If the normal build output is locked, verify the renderer in a fresh temporary
directory without replacing existing artifacts:

```powershell
$verificationDir = Join-Path $env:TEMP ('meeting-scribe-build-' + [guid]::NewGuid())
npm.cmd run build -- --outDir "$verificationDir"
```

## Notes For Agents

- Apply the Python test-construction rules in
  [PYTHON_STANDARDS.md](PYTHON_STANDARDS.md#tests): isolated resources, scoped
  patches, event/barrier synchronization, controlled clocks, and failure-path
  ownership assertions. Keep the existing `unittest` layout and runners.
- Prefer focused tests while iterating, then run `npm test` before finalizing broad changes.
- Do not use `--runInBand`; that is a Jest flag and Vitest rejects it.
- Avoid generating coverage unless the user asks for coverage or the task is coverage-related.
- Do not inspect `coverage/`, `node_modules/`, `.local/`, `models/`, or build artifacts unless the task specifically needs them.

## Metarhia Convergence Regressions

```powershell
npm run test:frontend -- frontend/electron/tests/python-backend.test.js frontend/electron/tests/main.test.js frontend/renderer/tests/app/useMeetingScribeApp/actions.test.jsx
```

```powershell
python -m unittest tests.backend.interface.test_jsonl_bridge_limits tests.backend.model_downloads.test_llm_model_transfer tests.backend.asr.test_worker_error_contract tests.backend.audio.test_wasapi_bindings -v
```

These checks cover message envelopes, request admission and deadlines, process
replacement, proxy isolation, repository authentication and offline behavior,
ASR error boundaries, and native callback completion without physical devices.
Session interface tests use an event signal instead of polling with sleeps.
