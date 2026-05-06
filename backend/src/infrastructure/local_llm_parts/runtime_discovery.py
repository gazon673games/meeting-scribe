from __future__ import annotations

import os
import shutil
from pathlib import Path

from infrastructure.local_llm_parts.errors import LocalLlmError


def find_llama_server(project_root: Path) -> Path:
    env_path = str(os.environ.get("LLAMA_SERVER_EXE", "")).strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(sorted((project_root / ".local" / "llama_cpp").glob("**/llama-server.exe")))
    candidates.extend(sorted((project_root / ".local" / "llama_cpp").glob("**/llama-server")))
    if found := shutil.which("llama-server"):
        candidates.append(Path(found))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise LocalLlmError(
        code="local_llm_server_missing",
        message="llama-server executable was not found",
        suggestion="Install llama.cpp locally or set LLAMA_SERVER_EXE to llama-server.",
    )


def find_direct_gguf_path(text: str, project_root: Path) -> Path | None:
    direct = Path(text).expanduser()
    candidates = [direct] if direct.is_absolute() else [direct, project_root / direct]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".gguf":
            return candidate.resolve()
    return None


def find_gguf_model(project_root: Path, model: str) -> Path:
    text = str(model or "").strip()
    if not text:
        raise LocalLlmError(
            code="model_required",
            message="Local OpenAI-compatible profile requires a model name.",
            suggestion="Select a GGUF model in Settings > Models > Language Models.",
        )

    direct = find_direct_gguf_path(text, project_root)
    if direct is not None:
        return direct

    wanted = Path(text).stem if text.lower().endswith(".gguf") else text
    models_root = project_root / "models" / "llm"
    matches = [
        path
        for path in sorted(models_root.rglob("*.gguf"), key=lambda item: str(item).lower())
        if path.stem == wanted or path.name == text
    ]
    if matches:
        return matches[0].resolve()

    raise LocalLlmError(
        code="local_llm_model_missing",
        message=f"GGUF model '{text}' was not found",
        suggestion="Download or choose a GGUF model from Settings > Models > Language Models.",
    )
