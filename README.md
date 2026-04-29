# Meeting Scribe

Meeting Scribe is a desktop application for real-time transcription of meetings, interviews, calls, and other spoken sessions. It is designed for users who need a readable live transcript, optional local recording, and quick assistant-generated summaries without manually taking detailed notes during the conversation.

The application uses an Electron + React interface and a local Python backend. Audio capture, speech recognition, transcript state, and assistant orchestration run on the user's machine.

## Purpose

Meeting Scribe helps turn live audio into structured working material:

- capture microphone and system audio in one desktop app;
- transcribe speech in real time with `faster-whisper`;
- keep a live transcript with source labels and speaker-friendly formatting;
- optionally save the captured audio for review or offline processing;
- generate assistant responses, summaries, action items, and risk checks from the current transcript context.

Typical use cases include interviews, team meetings, research calls, support sessions, demos, and any workflow where the user needs to listen actively while preserving a useful written record.

## Architecture

- `frontend/electron/` - Electron main process, preload bridge, and desktop assets.
- `frontend/renderer/` - React renderer UI.
- `backend/main_electron_backend.py` - Python backend entrypoint used by Electron.
- `backend/src/` - audio, ASR, transcript, assistant, and application logic.
- `models/` - local ASR/model cache.
- `tests/` - automated tests.
- `tools/` - release and utility scripts.

Electron owns the desktop window and UI lifecycle. The Python backend owns audio capture, transcription, and assistant commands. The two parts communicate through the Electron bridge, so the interface can evolve separately from the backend logic.

## Run Locally

Install dependencies:

```powershell
npm install
```

Start the full desktop application:

```powershell
npm run dev
```

Runtime files are stored in `.local/`. Model and cache files are stored in `models/`.

## Assistant Providers

Assistant profiles can use Codex CLI or local LLM runtimes:

- `codex` - Codex CLI with the configured command and optional proxy.
- `ollama` - Ollama HTTP API, default `http://127.0.0.1:11434`.
- `openai_local` - OpenAI-compatible local API, default `http://127.0.0.1:1234/v1` for tools such as LM Studio or llama.cpp server.

Each profile can set its own model, base URL, temperature, and max token limit. Local providers use Python stdlib HTTP calls, so no extra Python package is required.

Useful development commands:

```powershell
npm run dev:renderer   # React renderer only
npm run dev:electron   # Electron shell, when Vite is already running
npm run build          # production renderer build
```

## Build

Build a local Windows package:

```powershell
npm run package:win
```

Release archives are built by GitHub Actions when a `v*` tag is pushed. The current release matrix is:

- Windows x64
- Linux x64
- macOS x64

## Notes 

medium priority
1. reaseach idea - add support for diffrent ai assistent mode that use seleium to interact with deepseek
(to avoid api, off the top of my head it needs to use selenium and profiel with authorized deepseek acc, it might be the way that reduce delay even more that current that use fast model of codex )
2. fix frontend issues
3. optimization

low priority
1. review, goal - find all places where we shoud make refactoring by ddd aprouch and clean code princilples
2. rewirte readmi - add architecture, video
