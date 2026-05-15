from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHILD_ENV = "MEETING_SCRIBE_COVERAGE_CHILD"
SRC_ROOT = ROOT / "backend" / "src"
TOOLS_ROOT = ROOT / "tools"


def _venv_python() -> Path | None:
    candidates = (
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _run_in_venv_if_available() -> int | None:
    if os.environ.get(CHILD_ENV):
        return None
    python = _venv_python()
    if python is None or python.resolve() == Path(sys.executable).resolve():
        return None
    env = dict(os.environ)
    env[CHILD_ENV] = "1"
    return subprocess.run([str(python), str(Path(__file__).resolve())], cwd=ROOT, env=env, check=False).returncode


def _prepare_import_path() -> None:
    src_text = str(SRC_ROOT)
    tools_text = str(TOOLS_ROOT)
    sys.path[:] = [item for item in sys.path if str(Path(item or ".").resolve()) != tools_text]
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    for package_name in ("asr", "audio"):
        module = sys.modules.get(package_name)
        module_file = str(getattr(module, "__file__", "") or "")
        if module_file and module_file.startswith(tools_text):
            del sys.modules[package_name]


def main() -> int:
    redirected = _run_in_venv_if_available()
    if redirected is not None:
        return redirected

    _prepare_import_path()
    try:
        import coverage
    except ImportError:
        print("coverage.py is not installed. Run: python -m pip install -r requirements/requirements-dev.txt", file=sys.stderr)
        return 2

    cov = coverage.Coverage(config_file=str(ROOT / ".coveragerc"))
    cov.erase()
    cov.start()
    suite = unittest.defaultTestLoader.discover(start_dir=str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    cov.stop()
    cov.save()
    cov.report()
    cov.html_report()
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
