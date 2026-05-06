from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from application.asr_session import ASRSessionSettings
from application.local_paths import project_local_root
from interface.session_controller_parts.artifacts_identity import IdentityProcessor
from interface.session_controller_parts.helpers import clean_speaker, optional_float, safe_float


def resolve_app_data_dir(project_root: Path, settings: Optional[ASRSessionSettings]) -> Path:
    if settings is not None and str(settings.speaker_identity_app_data_dir or "").strip():
        return Path(str(settings.speaker_identity_app_data_dir)).expanduser().resolve()
    return project_local_root(project_root)


def final_transcript_lines(lines: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for raw in lines:
        if str(raw.get("id", "")).startswith("streaming-"):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        line = dict(raw)
        line["text"] = text
        line["stream"] = str(line.get("stream") or "mix")
        line["speaker"] = clean_speaker(line.get("speaker", "")) or str(line.get("stream") or "mix")
        line["ts"] = safe_float(line.get("ts", time.time()), time.time())
        line["t_start"] = optional_float(line.get("t_start"))
        line["t_end"] = optional_float(line.get("t_end"))
        out.append(line)
    out.sort(key=lambda line: safe_float(line.get("ts", 0.0), 0.0))
    return out


def build_session_speakers(lines: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    speakers: list[Dict[str, Any]] = []
    by_key: Dict[str, Dict[str, Any]] = {}
    next_index = 1
    for line in lines:
        stream = str(line.get("stream") or "mix")
        label = str(line.get("speaker") or stream)
        key = f"{stream}\x1f{label}"
        speaker = by_key.get(key)
        if speaker is None:
            speaker = {
                "session_speaker_id": f"spk_{next_index:03d}",
                "stream": stream,
                "label": label,
                "total_speech_ms": 0,
            }
            next_index += 1
            by_key[key] = speaker
            speakers.append(speaker)
        speaker["total_speech_ms"] = int(speaker.get("total_speech_ms") or 0) + line_duration_ms(line)
        line["session_speaker_id"] = speaker["session_speaker_id"]
    return speakers


def line_duration_ms(line: Dict[str, Any]) -> int:
    t_start = optional_float(line.get("t_start"))
    t_end = optional_float(line.get("t_end"))
    if t_start is None or t_end is None:
        return 0
    duration = max(0.0, float(t_end) - float(t_start))
    return int(round(duration * 1000.0))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_srt(path: Path, lines: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not lines:
        path.write_text("", encoding="utf-8")
        return
    base_ts = safe_float(lines[0].get("ts", 0.0), 0.0)
    prev_end = 0.0
    chunks: list[str] = []
    for index, line in enumerate(lines, start=1):
        t_start = optional_float(line.get("t_start"))
        t_end = optional_float(line.get("t_end"))
        ts = safe_float(line.get("ts", base_ts), base_ts)
        start = float(t_start) if t_start is not None else max(prev_end, ts - base_ts)
        end = float(t_end) if t_end is not None else start + 2.0
        if end <= start:
            end = start + 0.6
        prev_end = max(prev_end, end)
        speaker = str(line.get("speaker") or line.get("stream") or "mix")
        text = str(line.get("text") or "").strip()
        chunks.append(
            f"{index}\n"
            f"{srt_timestamp(start)} --> {srt_timestamp(end)}\n"
            f"{speaker}: {text}\n"
        )
    path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def srt_timestamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generated_session_id() -> str:
    return f"sess_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def session_speakers_payload(
    *,
    session_id: str,
    settings: Optional[ASRSessionSettings],
    embedding_model: str,
    speakers: list[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "session_id": session_id,
        "diarization_model": str(settings.diar_backend if settings else ""),
        "embedding_model": embedding_model,
        "speakers": [
            {
                "session_speaker_id": str(speaker.get("session_speaker_id") or ""),
                "label": str(speaker.get("label") or ""),
                "total_speech_ms": int(speaker.get("total_speech_ms") or 0),
                "embedding_id": speaker.get("embedding_id"),
                "matched_person_id": speaker.get("matched_person_id"),
                "match_similarity": float(speaker.get("match_similarity") or 0.0),
                "match_status": str(speaker.get("match_status") or "new"),
            }
            for speaker in speakers
        ],
    }


def write_session_outputs(
    *,
    asr: Any,
    settings: Optional[ASRSessionSettings],
    session_id: str,
    project_root: Path,
    transcript_lines: list[Dict[str, Any]],
) -> None:
    lines = final_transcript_lines(transcript_lines)
    resolved_session_id = str(session_id or generated_session_id())
    app_data_dir = resolve_app_data_dir(project_root, settings)
    session_dir = app_data_dir / "sessions" / resolved_session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    speakers = build_session_speakers(lines)
    identity = IdentityProcessor.from_runtime(
        session_id=resolved_session_id,
        asr=asr,
        settings=settings,
        app_data_dir=app_data_dir,
    )
    identity.apply_to_speakers(speakers)
    identity.apply_to_lines(lines, speakers)

    speakers_payload = session_speakers_payload(
        session_id=resolved_session_id,
        settings=settings,
        embedding_model=identity.embedding_model,
        speakers=speakers,
    )
    write_json(session_dir / "speakers.json", speakers_payload)
    write_jsonl(session_dir / "transcript.jsonl", lines)
    write_srt(session_dir / "transcript.srt", lines)
