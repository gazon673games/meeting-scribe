from __future__ import annotations

from typing import Any, Dict, Optional

from tools.batch_scribe_parts.request import BatchScribeRequest


def build_profile(
    profile_name: str = "Quality",
    *,
    model: str = "large-v3",
    device: str = "cuda",
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    vad_filter: bool = True,
    condition_on_previous_text: bool = True,
    compute_type: Optional[str] = None,
    beam_size: Optional[int] = None,
    cpu_threads: Optional[int] = None,
    num_workers: Optional[int] = None,
    temperature: Optional[float] = None,
    extra_transcribe_options: Optional[Dict[str, Any]] = None,
):
    from application.asr_profiles import profile_defaults  # type: ignore
    from asr.infrastructure.offline_runner import OfflineProfile  # type: ignore

    defaults = profile_defaults(profile_name)
    profile = OfflineProfile(
        model_name=model,
        device=device,
        compute_type=compute_type or defaults.get("compute_type", "float16"),
        beam_size=beam_size if beam_size is not None else defaults.get("beam_size", 6),
        language=language,
        initial_prompt=initial_prompt,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
    )
    profile.cpu_threads = cpu_threads  # type: ignore[attr-defined]
    profile.num_workers = num_workers  # type: ignore[attr-defined]
    profile.temperature = temperature  # type: ignore[attr-defined]
    profile.extra_transcribe_options = dict(extra_transcribe_options or {})  # type: ignore[attr-defined]
    return profile


def profile_from_request(request: BatchScribeRequest):
    return build_profile(
        request.profile_name,
        model=request.model,
        device=request.device,
        language=request.language,
        initial_prompt=request.initial_prompt,
        vad_filter=request.vad_filter,
        condition_on_previous_text=request.condition_on_previous_text,
        compute_type=request.compute_type,
        beam_size=request.beam_size,
        cpu_threads=request.cpu_threads,
        num_workers=request.num_workers,
        temperature=request.temperature,
        extra_transcribe_options=request.asr_options,
    )

