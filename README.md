# meeting-scribe

Desktop app for capturing system audio and microphone input, transcribing speech in real time, and showing live helper prompts during meetings or interviews.

## What it does

- captures Windows desktop audio via WASAPI loopback
- optionally captures microphone input
- mixes or splits sources for realtime ASR
- writes transcripts and session logs
- can run an offline transcription pass after recording stops
- includes a Codex helper panel for live prompts based on the current session log

## Run

```powershell
npm install
npm run dev
```

Runtime files are kept under `.local/`.
Model caches stay in `models/`.

The Python backend can be run directly for bridge/debug work:

```powershell
.venv\Scripts\python.exe main_electron_backend.py
```

`main.py` now defaults to the Electron dev shell.

## Electron UI

The Electron UI runs as a desktop shell and talks to the Python backend over
stdin/stdout.

Install the Electron/React dependencies and start the new desktop shell:

```powershell
npm install
npm run dev
```

If the Electron binary cannot be downloaded because GitHub is unavailable, the
installer may need explicit proxy support:

```powershell
$env:ELECTRON_GET_USE_PROXY = "1"
npm rebuild electron
```

The dev launcher can also use an extracted runtime under
`build/electron-runtime/`. It clears `ELECTRON_RUN_AS_NODE` for the app process
because that environment variable makes Electron behave like plain Node.

For renderer-only work, use:

```powershell
npm run dev:renderer
```

For the desktop shell only, with an already running renderer server:

```powershell
npm run dev:electron
```

The Electron bridge exposes backend health, config, device discovery, source
add/remove/toggle/delay controls, audio session start/stop, live ASR
startup/stop, ASR settings, WAV recording, transcript streaming/files,
assistant actions, and offline pass handling through headless controllers.

## Release build

Build a local Windows release archive:

```powershell
.\tools\release\build_exe.ps1 -Version dev
```

The release builder creates a portable Electron app, bundles the Python backend
as `resources\backend\meeting-scribe-backend.exe`, and writes
`dist\meeting-scribe-<version>-win64.zip`. Inside it, run `meeting-scribe.exe`.
If a developer shell has `ELECTRON_RUN_AS_NODE=1` set, use
`meeting-scribe.cmd`; it clears that variable before starting the same app.

GitHub Actions builds release archives on `v*` tags for Windows x64/arm64,
Linux x64/arm64, and macOS x64/arm64, then attaches them to the GitHub Release.
The build script runs the packaged Python backend with `--repair-config` and
`--smoke-import` before zipping the release.
Windows arm64 is currently allowed to fail because `faster-whisper` depends on
`ctranslate2`, which does not publish a `win_arm64` wheel for this dependency set yet.

Publish a release by pushing a version tag:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

Minimal CLI capture example:

```powershell
.venv\Scripts\python.exe -m tools.audio.capture_cli
```

## Notes

- built for Windows audio capture workflows
- ASR uses `faster-whisper`
- UI uses Electron/React
