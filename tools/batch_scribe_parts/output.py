from __future__ import annotations

import json
from pathlib import Path


def srt_ts(t: float) -> str:
    total_ms = int(round(max(0.0, t) * 1000))
    total_seconds, milliseconds = divmod(total_ms, 1000)
    total_minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(total_minutes, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}'


def write_output(segs: list[dict], out_path: Path, fmt: str) -> None:
    """Write segments to SRT, TXT, or JSONL."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == 'txt':
        lines = []
        for seg in segs:
            spk = seg.get('speaker', '')
            lines.append((f'[{spk}] ' if spk else '') + seg['text'])
        out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        return

    if fmt == 'jsonl':
        with out_path.open('w', encoding='utf-8') as f:
            for seg in segs:
                f.write(json.dumps(seg, ensure_ascii=False) + '\n')
        return

    blocks: list[str] = []
    for i, seg in enumerate(segs, 1):
        spk = seg.get('speaker', '')
        prefix = f'[{spk}] ' if spk else ''
        blocks.append(
            f'{i}\n'
            f'{srt_ts(seg["t0"])} --> {srt_ts(seg["t1"])}\n'
            f'{prefix}{seg["text"]}'
        )
    out_path.write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')


def append_srt_segment(seg: dict, index: int, f) -> None:
    spk = seg.get('speaker', '')
    prefix = f'[{spk}] ' if spk else ''
    block = (
        f'{index}\n'
        f'{srt_ts(seg["t0"])} --> {srt_ts(seg["t1"])}\n'
        f'{prefix}{seg["text"]}\n'
    )
    if index > 1:
        f.write('\n')
    f.write(block)
    f.flush()
