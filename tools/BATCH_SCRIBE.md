# Batch Scriber

`tools/batch_scribe.py` runs Meeting Scribe transcription outside the Electron UI.
It accepts media files that `ffmpeg` can read and writes `srt`, `txt`, or `jsonl`.

## Basic Usage

From the repository root:

```powershell
python tools/batch_scribe.py meeting.mp4
```

This writes `meeting.srt` next to the input file.

Write outputs to a directory:

```powershell
python tools/batch_scribe.py recordings\*.mp4 --out-dir out
```

Choose the output format:

```powershell
python tools/batch_scribe.py meeting.mp4 --format srt
python tools/batch_scribe.py meeting.mp4 --format txt
python tools/batch_scribe.py meeting.mp4 --format jsonl
```

## ASR Profiles, Models, And Language

Use the same profile names as the app:

```powershell
python tools/batch_scribe.py meeting.mp4 --profile Quality
python tools/batch_scribe.py meeting.mp4 --profile Realtime
python tools/batch_scribe.py meeting.mp4 --profile "Ultra Fast"
python tools/batch_scribe.py meeting.mp4 --profile Custom
```

Pick a model and language:

```powershell
python tools/batch_scribe.py meeting.mp4 --model large-v3 --language ru
python tools/batch_scribe.py meeting.mp4 --model small --language en
python tools/batch_scribe.py meeting.mp4 --model bzikst/faster-whisper-large-v3-russian --language ru
```

Use CPU:

```powershell
python tools/batch_scribe.py meeting.mp4 --device cpu --compute-type int8
```

## One-Time Advanced ASR Settings

These settings affect only the current batch run. They do not rewrite `config.json`
or app settings.

```powershell
python tools/batch_scribe.py meeting.mp4 --beam-size 8 --compute-type float16
python tools/batch_scribe.py meeting.mp4 --cpu-threads 8 --num-workers 2
python tools/batch_scribe.py meeting.mp4 --temperature 0.2
python tools/batch_scribe.py meeting.mp4 --no-vad
python tools/batch_scribe.py meeting.mp4 --no-condition-on-previous-text
```

Pass a raw faster-whisper transcribe option:

```powershell
python tools/batch_scribe.py meeting.mp4 --asr-option patience=1.5 --asr-option best_of=3
```

Values are coerced from strings where possible: `true`, `false`, integers, floats,
`none`, and `null`.

## Word-By-Word Output

Write one SRT block per recognized word when faster-whisper returns word
timestamps:

```powershell
python tools/batch_scribe.py meeting.mp4 --word-by-word --format srt
```

JSONL keeps the word records structured:

```powershell
python tools/batch_scribe.py meeting.mp4 --word-by-word --format jsonl
```

## Speaker Diarization

Run without diarization:

```powershell
python tools/batch_scribe.py meeting.mp4 --profile Quality
```

Run with the default local online diarizer:

```powershell
python tools/batch_scribe.py meeting.mp4 --diar
```

Choose a diarization backend:

```powershell
python tools/batch_scribe.py meeting.mp4 --diar --diar-backend online
python tools/batch_scribe.py meeting.mp4 --diar --diar-backend nemo
python tools/batch_scribe.py meeting.mp4 --diar --diar-backend sherpa_onnx --sherpa-model-path models\speaker.onnx
python tools/batch_scribe.py meeting.mp4 --diar --diar-backend pyannote
```

Tune the clustering threshold:

```powershell
python tools/batch_scribe.py meeting.mp4 --diar --diar-threshold 0.78
```

Use a separate diarization device:

```powershell
python tools/batch_scribe.py meeting.mp4 --diar --diar-device cpu
```

## Model Cache And Missing Models

By default the script configures the same project-local model cache used by the
desktop app: `models/`.

Use a custom cache directory:

```powershell
python tools/batch_scribe.py meeting.mp4 --models-dir D:\models\meeting-scribe
```

If a model is not present, faster-whisper may download it when network access is
available. If the machine is offline, the local path is invalid, or the cache is
missing the model, Batch Scriber raises a short actionable error:

```text
Unable to load ASR model '<name>'. The model may be missing from the local cache...
```

Fixes:

- Download the model in the app first.
- Run once with network access so faster-whisper can populate the cache.
- Pass a valid local model path as `--model`.
- Point `--models-dir` at the cache that already contains the model.

## Library Entry Point

Use `scribe_to_srt` for a single local file:

```python
from pathlib import Path
from tools.batch_scribe import scribe_to_srt

result = scribe_to_srt(
    Path("meeting.mp4"),
    Path("meeting.srt"),
    profile_name="Quality",
    model="large-v3",
    language="ru",
    word_by_word=True,
)

print(result.output_path)
```

Use `BatchScribeRequest` when you need an explicit request object:

```python
from pathlib import Path
from tools.batch_scribe import BatchScribeRequest, BatchScriber

request = BatchScribeRequest(
    input_path=Path("meeting.mp4"),
    output_path=Path("meeting.srt"),
    profile_name="Custom",
    model="small",
    device="cpu",
    compute_type="int8",
    beam_size=2,
    diar=True,
)

result = BatchScriber(request).run()
```

