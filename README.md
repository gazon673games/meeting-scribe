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

Minimal CLI capture example:

```powershell
.venv\Scripts\python.exe capture_cli.py
```

## Notes

- built for Windows audio capture workflows
- ASR uses `faster-whisper`
- UI uses `PySide6`
