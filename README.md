# Meeting Scribe

Desktop app for live meeting/interview transcription. The UI is Electron + React; the backend is Python and runs locally.

## Features

- microphone and system-audio capture
- realtime ASR with `faster-whisper`
- live transcript with per-source labels
- optional WAV recording and offline pass
- Codex assistant actions based on the current transcript
- portable release archives for Windows, Linux, and macOS x64

## Run Locally

```powershell
npm install
npm run dev
```

Runtime files are stored in `.local/`. Model/cache files stay in `models/`.

Renderer only:

```powershell
npm run dev:renderer
```

Electron shell only, when Vite is already running:

```powershell
npm run dev:electron
```

## Build

Windows local package:

```powershell
npm run package:win
```

Cross-platform releases are built by GitHub Actions when a `v*` tag is pushed. The current release matrix is:

- Windows x64
- Linux x64
- macOS x64

## Project Layout

- `frontend/electron/` - Electron main/preload code and desktop assets
- `frontend/renderer/` - React UI
- `backend/main_electron_backend.py` - Electron backend entrypoint
- `backend/src/` - Python app code
- `tests/` - Python tests
- `tools/` - release and utility scripts
