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
.venv\Scripts\python.exe main.py
```

## Release build

Build a local Windows release archive:

```powershell
.\scripts\build_exe.ps1 -Version dev
```

The archive is written to `dist\meeting-scribe-<version>-win64.zip`.
Inside it, run `meeting-scribe.exe`.

GitHub Actions builds release archives on `v*` tags for Windows x64/arm64,
Linux x64/arm64, and macOS x64/arm64, then attaches them to the GitHub Release.
The build script also runs the packaged app with `--smoke-import` before zipping the release.
Windows arm64 is currently allowed to fail because `faster-whisper` depends on
`ctranslate2`, which does not publish a `win_arm64` wheel for this dependency set yet.

Publish a release by pushing a version tag:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

Minimal CLI capture example:

```powershell
.venv\Scripts\python.exe capture_cli.py
```

## Notes

- built for Windows audio capture workflows
- ASR uses `faster-whisper`
- UI uses `PySide6`
