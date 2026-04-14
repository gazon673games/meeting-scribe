from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
import zipfile
from pathlib import Path


APP_NAME = "meeting-scribe"


def _safe_token(value: str, default: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip())
    return token or default


def _default_artifact_suffix() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"arm64", "aarch64"} else "x64"
    if system == "darwin":
        return f"macos-{arch}"
    if system == "linux":
        return f"linux-{arch}"
    if system == "windows":
        return f"windows-{arch}"
    return f"{system or 'unknown'}-{arch}"


def _app_executable(dist_dir: Path) -> Path:
    exe_name = f"{APP_NAME}.exe" if platform.system().lower() == "windows" else APP_NAME
    return dist_dir / exe_name


def _zip_dir_contents(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


def build_release(version: str, artifact_suffix: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    spec_path = repo_root / "packaging" / "meeting_scribe.spec"
    dist_dir = repo_root / "dist" / APP_NAME
    exe_path = _app_executable(dist_dir)
    zip_path = (
        repo_root
        / "dist"
        / f"{APP_NAME}-{_safe_token(version, 'manual')}-{_safe_token(artifact_suffix, _default_artifact_suffix())}.zip"
    )

    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec_path)],
        cwd=repo_root,
        check=True,
    )

    if not dist_dir.exists():
        raise RuntimeError(f"Expected PyInstaller output was not found: {dist_dir}")
    if not exe_path.exists():
        raise RuntimeError(f"Expected executable was not found: {exe_path}")

    subprocess.run([str(exe_path), "--repair-config"], cwd=dist_dir, check=True)
    subprocess.run([str(exe_path), "--smoke-import"], cwd=dist_dir, check=True)

    _zip_dir_contents(dist_dir, zip_path)
    print(f"Built {zip_path}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PyInstaller release archive.")
    parser.add_argument("--version", default="dev")
    parser.add_argument("--artifact-suffix", default=_default_artifact_suffix())
    args = parser.parse_args()
    build_release(version=args.version, artifact_suffix=args.artifact_suffix)


if __name__ == "__main__":
    main()
