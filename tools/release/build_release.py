from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


APP_NAME = "meeting-scribe"
BACKEND_NAME = "meeting-scribe-backend"


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


def _executable_name(name: str) -> str:
    return f"{name}.exe" if platform.system().lower() == "windows" else name


def _app_executable(dist_dir: Path) -> Path:
    return dist_dir / _executable_name(APP_NAME)


def _backend_executable(backend_dir: Path) -> Path:
    return backend_dir / _executable_name(BACKEND_NAME)


def _zip_dir_contents(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


def _remove_tree_inside_dist(path: Path, repo_root: Path) -> None:
    resolved = path.resolve()
    dist_root = (repo_root / "dist").resolve()
    if resolved != dist_root and dist_root not in resolved.parents:
        raise RuntimeError(f"Refusing to remove path outside dist: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def _copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        raise RuntimeError(f"Required build input was not found: {source}")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _write_packaged_package_json(repo_root: Path, app_dir: Path) -> None:
    package = json.loads((repo_root / "package.json").read_text(encoding="utf-8"))
    packaged = {
        "name": package.get("name", APP_NAME),
        "version": package.get("version", "0.0.0"),
        "private": True,
        "main": package.get("main", "frontend/electron/main.cjs"),
    }
    (app_dir / "package.json").write_text(json.dumps(packaged, indent=2), encoding="utf-8")


def _write_windows_launcher(final_dir: Path) -> None:
    if platform.system().lower() != "windows":
        return
    launcher = "\r\n".join(
        [
            "@echo off",
            "setlocal",
            'set "ELECTRON_RUN_AS_NODE="',
            'start "" "%~dp0meeting-scribe.exe" %*',
            "",
        ]
    )
    (final_dir / "meeting-scribe.cmd").write_text(launcher, encoding="utf-8")


def _prepare_electron_app(repo_root: Path, final_dir: Path, backend_build_dir: Path) -> None:
    electron_dist = repo_root / "node_modules" / "electron" / "dist"
    renderer_build = repo_root / "build" / "electron-renderer"
    if not electron_dist.exists():
        raise RuntimeError("Electron runtime is missing. Run `npm install` or `npm rebuild electron` first.")
    if not renderer_build.exists():
        raise RuntimeError("Renderer build is missing. Run `npm run build` first.")

    _remove_tree_inside_dist(final_dir, repo_root)
    _copy_tree(electron_dist, final_dir)

    electron_exe = final_dir / _executable_name("electron")
    app_exe = _app_executable(final_dir)
    if electron_exe.exists() and electron_exe != app_exe:
        electron_exe.rename(app_exe)
    if not app_exe.exists():
        raise RuntimeError(f"Electron executable was not found after runtime copy: {app_exe}")

    resources_dir = final_dir / "resources"
    app_dir = resources_dir / "app"
    backend_dir = resources_dir / "backend"
    app_dir.mkdir(parents=True, exist_ok=True)

    _write_packaged_package_json(repo_root, app_dir)
    _copy_tree(repo_root / "frontend" / "electron", app_dir / "frontend" / "electron")
    _copy_tree(renderer_build, app_dir / "build" / "electron-renderer")
    _copy_tree(backend_build_dir, backend_dir)
    _write_windows_launcher(final_dir)


def build_release(version: str, artifact_suffix: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    spec_path = Path(__file__).resolve().with_name("meeting_scribe.spec")
    backend_build_dir = repo_root / "dist" / BACKEND_NAME
    final_dir = repo_root / "dist" / APP_NAME
    zip_path = (
        repo_root
        / "dist"
        / f"{APP_NAME}-{_safe_token(version, 'manual')}-{_safe_token(artifact_suffix, _default_artifact_suffix())}.zip"
    )

    npm = "npm.cmd" if platform.system().lower() == "windows" else "npm"
    subprocess.run([npm, "run", "build"], cwd=repo_root, check=True)
    subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", str(spec_path)],
        cwd=repo_root,
        check=True,
    )

    if not backend_build_dir.exists():
        raise RuntimeError(f"Expected PyInstaller backend output was not found: {backend_build_dir}")
    backend_exe = _backend_executable(backend_build_dir)
    if not backend_exe.exists():
        raise RuntimeError(f"Expected backend executable was not found: {backend_exe}")

    _prepare_electron_app(repo_root, final_dir, backend_build_dir)

    env = dict(os.environ)
    env["MEETING_SCRIBE_RUNTIME_ROOT"] = str(final_dir)
    subprocess.run([str(_backend_executable(final_dir / "resources" / "backend")), "--repair-config"], cwd=final_dir, env=env, check=True)
    subprocess.run([str(_backend_executable(final_dir / "resources" / "backend")), "--smoke-import"], cwd=final_dir, env=env, check=True)

    if not _app_executable(final_dir).exists():
        raise RuntimeError(f"Expected Electron executable was not found: {_app_executable(final_dir)}")

    _zip_dir_contents(final_dir, zip_path)
    print(f"Built {zip_path}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable Electron release archive.")
    parser.add_argument("--version", default="dev")
    parser.add_argument("--artifact-suffix", default=_default_artifact_suffix())
    args = parser.parse_args()
    build_release(version=args.version, artifact_suffix=args.artifact_suffix)


if __name__ == "__main__":
    main()
