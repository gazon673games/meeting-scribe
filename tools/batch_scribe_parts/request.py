from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.batch_scribe_parts.constants import OUTPUT_FORMATS


def output_path_for(src: Path, fmt: str, out_dir: Optional[Path]) -> Path:
    ext = {"srt": ".srt", "txt": ".txt", "jsonl": ".jsonl"}[fmt]
    return (out_dir / (src.stem + ext)) if out_dir else src.with_suffix(ext)


@dataclass(frozen=True)
class BatchScribeRequest:
    """Single-use request for local batch transcription."""

    input_path: Path
    project_root: Optional[Path] = None
    models_dir: Optional[Path] = None
    output_path: Optional[Path] = None
    output_format: str = "srt"
    out_dir: Optional[Path] = None

    profile_name: str = "Quality"
    model: str = "large-v3"
    device: str = "cuda"
    language: Optional[str] = None
    initial_prompt: Optional[str] = None
    vad_filter: bool = True
    condition_on_previous_text: bool = True
    compute_type: Optional[str] = None
    beam_size: Optional[int] = None
    cpu_threads: Optional[int] = None
    num_workers: Optional[int] = None
    temperature: Optional[float] = None
    asr_options: Dict[str, Any] = field(default_factory=dict)

    word_by_word: bool = False

    diar: bool = False
    diar_backend: str = "online"
    diar_threshold: float = 0.74
    diar_device: Optional[str] = None
    sherpa_model_path: str = ""
    sherpa_provider: str = "cpu"
    sherpa_threads: int = 1

    def resolved_output_path(self) -> Path:
        fmt = str(self.output_format or "srt").lower()
        if fmt not in OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output format: {self.output_format}")
        if self.output_path is not None:
            return Path(self.output_path)
        out_dir = Path(self.out_dir) if self.out_dir is not None else None
        return output_path_for(Path(self.input_path), fmt, out_dir)


@dataclass(frozen=True)
class BatchScribeResult:
    input_path: Path
    output_path: Path
    output_format: str
    segments: List[dict]
