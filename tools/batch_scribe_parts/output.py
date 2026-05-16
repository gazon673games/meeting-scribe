from __future__ import annotations

import json
from pathlib import Path
from typing import List


def srt_ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_output(segs: List[dict], out_path: Path, fmt: str) -> None:
    """Write segments to SRT, TXT, or JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "txt":
        lines = []
        for seg in segs:
            spk = seg.get("speaker", "")
            lines.append((f"[{spk}] " if spk else "") + seg["text"])
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    if fmt == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for seg in segs:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
        return

    blocks: List[str] = []
    for i, seg in enumerate(segs, 1):
        spk = seg.get("speaker", "")
        prefix = f"[{spk}] " if spk else ""
        blocks.append(
            f"{i}\n"
            f"{srt_ts(seg['t0'])} --> {srt_ts(seg['t1'])}\n"
            f"{prefix}{seg['text']}"
        )
    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def append_srt_segment(seg: dict, index: int, f) -> None:
    spk = seg.get("speaker", "")
    prefix = f"[{spk}] " if spk else ""
    block = (
        f"{index}\n"
        f"{srt_ts(seg['t0'])} --> {srt_ts(seg['t1'])}\n"
        f"{prefix}{seg['text']}\n"
    )
    if index > 1:
        f.write("\n")
    f.write(block)
    f.flush()

