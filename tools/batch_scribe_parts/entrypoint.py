from __future__ import annotations

import tempfile
from pathlib import Path

from tools.batch_scribe_parts.output import write_output
from tools.batch_scribe_parts.profiles import profile_from_request
from tools.batch_scribe_parts.request import BatchScribeRequest, BatchScribeResult
from tools.batch_scribe_parts.scribe import Scribe


def configure_batch_cache(request: BatchScribeRequest) -> None:
    try:
        from application.local_paths import configure_project_local_io
    except Exception:
        return
    project_root = Path(request.project_root).resolve() if request.project_root else Path(__file__).resolve().parents[2]
    configure_project_local_io(project_root, models_dir=request.models_dir)


def scribe_kwargs_from_request(request: BatchScribeRequest) -> dict:
    return {
        "diar": request.diar,
        "diar_backend": request.diar_backend,
        "diar_sim_threshold": request.diar_threshold,
        "diar_device": request.diar_device,
        "sherpa_model_path": request.sherpa_model_path,
        "sherpa_provider": request.sherpa_provider,
        "sherpa_num_threads": request.sherpa_threads,
    }


class BatchScriber:
    """High-level local entry point that can later sit behind an API handler."""

    def __init__(self, request: BatchScribeRequest) -> None:
        self.request = request

    def run(self) -> BatchScribeResult:
        request = self.request
        input_path = Path(request.input_path)
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        configure_batch_cache(request)
        output_format = str(request.output_format or "srt").lower()
        output_path = request.resolved_output_path()
        profile = profile_from_request(request)

        with tempfile.TemporaryDirectory() as tmp:
            with Scribe(profile, **scribe_kwargs_from_request(request)) as scribe:
                segments = scribe.process(input_path, Path(tmp), word_by_word=request.word_by_word)

        write_output(segments, output_path, output_format)
        return BatchScribeResult(
            input_path=input_path,
            output_path=output_path,
            output_format=output_format,
            segments=segments,
        )


def scribe_to_srt(input_path: Path | str, output_path: Path | str | None = None, **params) -> BatchScribeResult:
    request = BatchScribeRequest(
        input_path=Path(input_path),
        output_path=Path(output_path) if output_path is not None else None,
        output_format="srt",
        **params,
    )
    return BatchScriber(request).run()
