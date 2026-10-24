# Project Map

## Top Level

- `backend/` - Python backend runtime, application logic, ASR, diarization, persistence, and Electron bridge.
- `frontend/electron/` - Electron main process, preload API, IPC handlers, dev launcher, and app assets.
- `frontend/renderer/` - React UI and client-side state management.
- `tests/backend/` - Python unit and integration-style backend tests grouped by area.
- `tests/helpers/` - shared Python test helpers.
- `tools/` - Release, coverage, and helper scripts.
- `requirements/` - Python dependency files.
- `config.json` - Local runtime configuration used by the app.

Avoid using these for broad context unless the task requires it: `.local/`, `models/`, `node_modules/`, `build/`, `dist/`, `coverage/`, `.venv/`.

## Backend

### Entry Points

- `backend/main_electron_backend.py` - backend process entry for Electron.
- `backend/src/interface/jsonl_bridge.py` - JSONL request/event bridge.
- `backend/src/interface/backend_protocol.py` - request validation and response records, independent of streams and controllers.
- `backend/src/interface/backend_parts/session_dispatch.py` - immutable API method definitions, invocation, and lifecycle execution policy.
- `backend/src/shared/domain/mappings.py` - shared parameter alias reader; distinguishes absent keys from present null, false, zero, and empty values.
- `backend/src/interface/backend_impl.py` - backend state ownership and explicit module operation composition, without mixin inheritance.
- `backend/src/interface/backend_parts/` - request handlers for state, sessions, models, downloads, and orchestration.

### Session And Transcript

- `backend/src/interface/session_controller.py` - headless session facade with explicit module operation bindings and lazy recorder startup.
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
- `backend/src/application/llm_model_sources.py` - source records, parsing, and deterministic GGUF selection.
- `backend/src/infrastructure/llm_model_transfer.py` - HTTP transfer and repository metadata with per-operation proxy configuration.
- `backend/src/interface/backend_parts/model_state_mixin.py` - model/download state and selected-model checks.

## Frontend

### Electron

- `frontend/electron/main.cjs` - Electron bootstrap with explicit runtime dependencies.
- `frontend/electron/desktop-runtime.cjs` - desktop composition factory, windows, and lifecycle state.
- `frontend/electron/python-backend.cjs` - backend process and JSONL transport, with injected process, filesystem, and lifecycle dependencies.
- `frontend/electron/backend-protocol.cjs` - pure numeric conversion and stopped-request response helpers.
- `frontend/electron/resource-monitor.cjs` - CPU, memory, and cached GPU snapshots, with injected runtime dependencies.
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

- Transcript matching: `tests/backend/domain/test_transcript_lines.py`
- Session transcript/speaker updates: `tests/backend/interface/test_electron_interface_session.py`
- Diarization sidecar: `tests/backend/diarization/test_diarization_updates.py`
- Diarization runtime: `tests/backend/diarization/test_diarization_runtime.py`
- Electron IPC/preload: `frontend/electron/tests/`
- React app flow: `frontend/renderer/tests/app/App.flow.test.jsx`
- Local LLM: `tests/backend/assistant/test_local_llm.py`, `tests/backend/assistant/test_local_llm_runtime_parts.py`

## Metarhia Architecture Decisions

The reference checkout is `D:/work/other/metrarhia`. The relevant sources are:

- `metaskills/skills/js-conventions/SKILL.md`: small functions, stable object shapes, and explicit naming.
- `metaskills/skills/js-gof/SKILL.md`: composition, dependency injection, declarative dispatch, and separation of system and domain code.
- `metaskills/skills/error-handling/SKILL.md`: operational failures and error translation at API boundaries.
- `Docs/content/en/LAYERS.md` and `impress/lib/procedure.js`: API contracts and per-method concurrency policies.
- `metacom-nodejs-example/server.js` and `lib/loader.js`: explicit runtime assembly and method registries.
- `Example/application/domain/chat.js`: domain operations over explicit collections.
- `BestPractice/README.md`: Python is allowed for application code; React and child processes are supported choices.

### Runtime Choice

Keep Python for audio capture, ASR, diarization, and model execution. The current
requirements and adapters use faster-whisper, CTranslate2, NumPy, sherpa-onnx,
SoundCard, and sounddevice. The optional GigaAM environment also uses PyTorch,
torchaudio, and Transformers. A complete Node.js migration would require replacing
or wrapping these adapters and verifying model output, timing, devices, and GPU
behavior. That is a runtime migration, beyond a behavior-preserving refactor.

Keep Electron and React in JavaScript. Node.js owns the desktop lifecycle and
starts the Python process. Python owns application services and native model
execution. There is no demonstrated benefit from moving only Python orchestration
to Node.js while retaining the same Python model process.

Apply Metarhia principles through native Python tools: frozen dataclasses for
request and method records, an immutable mapping for the method registry, explicit
constructor dependencies, and separate transport and application functions.
Legacy mixin exports remain compatibility facades. Production backend and
session controllers compose typed module operations without mixin inheritance. No JavaScript VM, dynamic code loader, Impress server, or Metacom
network transport is needed for the existing desktop JSONL contract.

### Request And Event Boundaries

- A request keeps the existing `id`, non-empty string `method`, and object or null
  `params`. Missing `params` defaults to an empty object.
- Success keeps `{id, ok: true, result}`. Failure keeps
  `{id, ok: false, error: {type, message}}` and a traceback on stderr.
- `BackendRequest` validates messages. `JsonLineBridge` parses each line once,
  owns streams and executors, and translates exceptions into error responses.
- `BACKEND_METHODS` is the allowlist. Its records define parameter passing and
  serial execution. The exported method sets derive from these same records.
- Start and stop requests share one lifecycle executor. Other requests use the
  existing worker pool. Leaving the bridge drains both executors.
- The JavaScript transport factory receives `spawn`, `existsSync`, packaged-mode
  and shutdown callbacks, and a logger. It has no Electron import.
- Each resource monitor owns its GPU cache. Its public snapshot shape and cache
  duration remain unchanged.
- React uses fixed handler tables for backend events and existing pure state
  reducers. Event names, notices, refresh rules, and transcript updates stay the
  same. Lookups only accept own table entries.

`npm run lint` checks all first-party JavaScript and JSX, including tests,
Electron modules, renderer modules, and build/quality configuration, against
`eslint-config-metarhia` and Prettier. CSS shares the formatter. External
protocol field names retain snake_case. Prettier
owns expression parentheses so ESLint and Prettier produce consistent output.

### Python Implementation Preferences

The detailed project rules are in [PYTHON_STANDARDS.md](PYTHON_STANDARDS.md).
They combine the local Metarhia preferences with concrete Python examples from
`D:/work/other/standarts/Python`, including Trio, HTTPX/httpcore, Django,
aiohttp, Starlette, and Pallets projects. The document records source locations,
selection conditions, and the boundary between checked and reviewed rules.
The table below is a summary; apply the detailed rules when choosing types,
control flow, exception recovery, resource ownership, and test construction.

These rules apply when implementing or refactoring Python behavior. `pyproject.toml`
checks the entire `backend/` tree, Python scripts under `tools/`, the PyInstaller
specification, and `tests/`. Test changes require explicit authorization,
which was granted for the follow-up
review. The batch entry point `tools/batch_scribe.py` follows the same Python
checks as the other first-party sources.

| Prefer | Use the alternative when |
| --- | --- |
| Module functions for stateless work; dataclasses for fixed internal records | A class enforces an object's invariant or owns state, resources, a lifecycle, or a required Python protocol. A dict represents a dynamic mapping or wire payload. |
| A static mapping for fixed key-to-value or key-to-handler selection | Conditions express ranges, ordering, or business rules. Keep those conditions explicit. |
| Static definitions at module scope | Data depends on a request. Build request-specific data inside the function. |
| Named intermediate values for compound expressions | A short expression already states one clear operation. Avoid nested conditional expressions. |
| Guard returns and `continue` for rejected cases | The branches form a short, symmetric decision. Avoid deep nesting. |
| Comprehensions or generators for short, pure transformations | Processing has side effects, early exit, or several steps. Use an explicit loop; do not use a comprehension for side effects. |
| Keyword arguments for defaults, bounds, and flags | Positional arguments have an obvious role. Avoid unexplained sequences such as `0, 0, 64`. |
| Direct field access for known contracts | The key is actually dynamic, as in a registry. Avoid `getattr` as a substitute for a declared dependency. |
| New output records and local accumulators | Mutation is part of an explicitly owned object's contract. Do not mutate caller-owned inputs as a shortcut. |
| A returned copy of shared mutable defaults | The API explicitly promises a shared read-only view. Do not expose the profile table for modification. |
| `is None` when only absence triggers a fallback | Empty strings, zero, or false must also trigger it. Preserve each existing fallback rule during a style refactor. |
| Existing helpers for aliases and normalization | The existing helper has different semantics. Preserve alias priority and distinguish missing values from present null or false values. |
| Built-in collection annotations such as `dict[str, Any]` and `T \| None` | Runtime compatibility requires an older syntax. The checked modules target Python 3.10 or newer. |

The Python reference review adds these concrete choices:

- Use `dataclass` for a fixed internal schema, `TypedDict` for a fixed mapping
  contract, and a small `Protocol` for an injected capability. Use a callable
  annotation for a dependency with one operation. Keep JSON dictionaries at
  serialization boundaries and preserve their external field names.
- Use `frozen=True` for value records and snapshots with stable fields;
  owned runtime state stays mutable. Check nested ownership separately.
  `slots=True` requires compatibility with inheritance, serialization and
  dynamic attributes; it is not an automatic change to every class.
- Select `list`, `tuple`, `dict`, `set`, `frozenset`, or `deque` by ordering,
  multiplicity, indexing, mutability and consumption requirements. A small
  collection size alone does not determine the Python container.
- Use a sentinel when missing, `None`, zero and false have different meanings.
  Read required keys directly after boundary validation; `.get()` expresses
  an optional field with a defined default.
- Pair resource acquisition with `with`, `try/finally`, or an `ExitStack`.
  Protect partial startup and transfer ownership only after it succeeds.
- Use short locks for shared state transitions. Perform detached cleanup and
  joins outside the state lock; serialize access to an active native handle
  with its own lock when its contract requires it.
- Give background work an owner, a completion path and a stop signal tied to
  that run. Use monotonic time for timeouts, cache expiry and durations.
- Keep expected recovery narrow. A broad worker or transport boundary reports
  unexpected failures; cancellation cleanup restores invariants and propagates
  cancellation. Preserve exception causes when translating operational errors.
- Test observable behavior and failure cleanup. Coordinate races with events
  or barriers, control timeout clocks, restore patches and isolate temporary
  resources. A synchronous thread fake verifies sequential logic only.

Use `snake_case` for Python names, descriptive verbs for functions, boolean
prefixes such as `is_` and `has_`, and unit suffixes such as `_ms` and `_s`.
Protocol field names keep their external spelling. Use four spaces, single
quotes, trailing commas in expanded records, and an 80-column formatting target.

Concrete applications:

- `asr/domain/profiles.py` declares profile settings as data and returns a copy
  through one lookup, instead of returning a record from each conditional branch.
- `session_controller_parts/helpers.py` declares source aliases and speaker
  identity aliases once. Request processing only reads those definitions.
- `session_controller_parts/settings.py` uses `first_param` instead of nested
  `params.get` chains. Bounds and defaults use named arguments. The existing
  precedence, coercion, and null handling remain unchanged.
- Streaming, diarization, and speaker identity settings are built by separate
  functions. State snapshots use separate configuration and options builders.
- Model ordering, policy modes, language prompts, source speaker labels, and
  assistant actions use static tables. Table values retain their original
  order, and mutable configuration results remain independent copies.
- The shared `first_param` implementation is reused by request, configuration,
  and provider adapters. Its original import path remains available.
- Best-effort cleanup failures retain their existing recovery behavior and
  emit diagnostic messages through module loggers on stderr when enabled.

### Full Python Review

The file inventory and decisions are recorded in `python_refactor_audit.json`.
The review covers all production Python files and helper scripts in scope,
including modules that only contain records, protocols, constants, or exports.
No code change is required merely because a file was reviewed.

Public call signatures, legacy facade imports, JSONL messages, model order,
fallback values, queue bounds, and audio processing algorithms remain compatible.
Existing stateful runtime classes, domain state machines, and native API
adapters remain explicit. Optional third-party capability checks and the order
of ctypes, NumPy compatibility, and application bootstrap operations are kept.

`npm run quality` checks style, dependency boundaries, and the existing complexity
budget. Python style checks cover the full scope. Test changes follow the explicit authorization rule in `AGENTS.md`. Hardware execution and native model quality require their normal
device and model environments; static checks and unit tests do not replace them.

Ruff checks formatting, imports, basic correctness, unnecessary collection
construction, return structure, and supported annotation syntax. The preference
for a table, a loop, or a function still requires code review; it is not a rule
to replace every conditional or class mechanically. JavaScript-specific V8
optimization rules do not imply equivalent Python performance guarantees.

### Follow-Up Convergence Review

The file-by-file decisions for the 17 requested files are recorded in
`METARHIA_REFACTOR.md`. Production facades now use explicit function bindings;
legacy mixin class names only preserve import compatibility. Operations receive
a typed owning facade instead of inheriting undeclared state through an MRO.

Desktop runtime construction accepts dependencies directly. Shutdown state stays
inside the runtime, and tests call its factory without global bootstrap hooks.
Both transports bound pending work. The JavaScript transport validates response
and event envelopes, cleans up deadlines and write errors, and ignores events
from a replaced process.

Local model start/stop functions implement an injected port. Optional provider
capabilities are normalized once at the compatibility boundary. HTTP model
transfers use a proxy opener for each operation; repository metadata retains
Hugging Face SDK authentication and offline behavior.

Native ABI declarations and activation callbacks are separate from audio source
lifecycle code. Named activation records retain the callback lifetime references.
VAD scenario recipes are static data, and metric aggregation is separate from
CLI orchestration. Renderer hook and action files use the shared JS style checks.

### Whole Project Follow-up

The separate continuous manual review is tracked in `METARHIA_LINE_REVIEW.md`
and `metarhia_line_review.json`, including pending ranges and open findings.
Earlier inventories combine static checks with focused source review and do
not establish that every line was manually inspected.

See `METARHIA_PROJECT_AUDIT.md` for the earlier scope, verification, and deliberate
exceptions. `metarhia_project_audit.json` contains one disposition per file.
The older Python inventory and convergence report describe their own snapshots.

- ASR and assistant fallback policies are module functions implementing typed
  policy ports. Legacy class imports remain compatibility facades.
- `infrastructure/http_transport.py` opens LLM and diarization HTTP requests
  with per-operation proxy configuration.
- `infrastructure/huggingface_transfer.py` passes proxy settings to SDK calls
  when supported. The environment-only compatibility path retains a lock and
  restoration of previous values.
- `FileTranscriptStore` owns typed pending SRT records and releases its file
  handles even when flushing fails.
- Preload exposes its API through an explicit dependency boundary; startup
  contains no global test dependency override.
- Diagnostics dispatch tables accept only their own handlers. Unknown event
  names cannot invoke inherited object properties.
- Context-free JavaScript functions use arrows with helpers before consumers;
  module initialization order is preserved. Generators and receiver methods
  retain their required forms.
- Model policy consumes `CodexProfile` records directly. An ordered model index
  preserves catalog order for the final fallback after recommendation priority.
- Bundled configuration recovery logs expected I/O and decoding failures;
  programming errors propagate from read, seed, and merge operations.
