# --- File: D:\work\own\voice2textTest\asr\offline_runner.py ---
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OfflineProfile:
    model_name: str = "large-v3"
    device: str = "cuda"             # "cuda" or "cpu"
    compute_type: str = "float16"    # "float16" / "int8_float16" / etc
    beam_size: int = 6
    language: Optional[str] = None   # None => auto
    initial_prompt: Optional[str] = None

    vad_filter: bool = True
    condition_on_previous_text: bool = True

    # faster-whisper defaults are fine; keep knobs minimal


class OfflineRunner:
    """
    Offline pass over a saved WAV for higher quality transcript.
    Writes:
      - <out_txt>: plain text transcript
      - <out_jsonl> (optional): structured segments
    """

    def __init__(self, *, project_root: Path):
        self.project_root = Path(project_root)

    def run(
        self,
        wav_path: Path,
        *,
        out_txt: Path,
        out_jsonl: Optional[Path] = None,
        profile: Optional[OfflineProfile] = None,
    ) -> Path:
        profile = profile or OfflineProfile()
        wav_path = Path(wav_path)
        out_txt = Path(out_txt)
        if out_jsonl is not None:
            out_jsonl = Path(out_jsonl)

        out_txt.parent.mkdir(parents=True, exist_ok=True)
        if out_jsonl is not None:
            out_jsonl.parent.mkdir(parents=True, exist_ok=True)

        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Offline pass requires faster-whisper.\n"
                "Install:\n"
                "  pip install faster-whisper\n"
                f"Import error: {type(e).__name__}: {e}"
            )

        model = WhisperModel(
            profile.model_name,
            device=profile.device,
            compute_type=profile.compute_type,
        )

        t0 = time.time()
        segments, info = model.transcribe(
            str(wav_path),
            language=profile.language,  # None => auto
            beam_size=int(profile.beam_size),
            initial_prompt=profile.initial_prompt,
            vad_filter=bool(profile.vad_filter),
            condition_on_previous_text=bool(profile.condition_on_previous_text),
        )

        # write outputs
        txt_parts = []
        fj = None
        if out_jsonl is not None:
            fj = out_jsonl.open("a", encoding="utf-8")

        try:
            for seg in segments:
                s_text = (seg.text or "").strip()
                if not s_text:
                    continue
                txt_parts.append(s_text)

                if fj is not None:
                    rec = {
                        "type": "offline_segment",
                        "t0": float(seg.start),
                        "t1": float(seg.end),
                        "text": s_text,
                        "ts": time.time(),
                    }
                    import json

                    fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
        finally:
            if fj is not None:
                try:
                    fj.close()
                except Exception:
                    pass

        full_text = "\n".join(txt_parts).strip()
        out_txt.write_text(full_text + ("\n" if full_text else ""), encoding="utf-8")

        dt = time.time() - t0
        # return path to main transcript
        return out_txt
