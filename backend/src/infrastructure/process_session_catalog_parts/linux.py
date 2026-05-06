from __future__ import annotations

from typing import Any, Dict, List


def list_linux_sessions() -> List[Dict[str, Any]]:
    try:
        import re
        import subprocess

        result = subprocess.run(
            ["pactl", "list", "sink-inputs"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []

        sessions: Dict[str, Dict[str, Any]] = {}
        current: Dict[str, str] = {}
        current_index: str = ""

        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            match = re.match(r"^Sink Input #(\d+)$", line)
            if match:
                if current_index:
                    _merge_linux_session(sessions, current_index, current)
                current_index = match.group(1)
                current = {}
                continue
            match = re.match(r'application\.name\s*=\s*"(.+)"', line)
            if match:
                current["app_name"] = match.group(1)
                continue
            match = re.match(r'application\.process\.id\s*=\s*"(\d+)"', line)
            if match:
                current["pid"] = match.group(1)

        if current_index:
            _merge_linux_session(sessions, current_index, current)

        return [
            {
                "pid": int(value.get("pid", 0)),
                "label": value["app_name"],
                "streams": value["streams"],
                "index": int(value["first_index"]),
            }
            for value in sessions.values()
            if "app_name" in value
        ]
    except Exception:
        return []


def _merge_linux_session(
    sessions: Dict[str, Dict[str, Any]],
    index: str,
    current: Dict[str, str],
) -> None:
    if "app_name" not in current:
        return
    key = current.get("pid") or f"idx_{index}"
    if key not in sessions:
        sessions[key] = {
            "app_name": current["app_name"],
            "pid": current.get("pid", "0"),
            "streams": 0,
            "first_index": index,
        }
    sessions[key]["streams"] += 1
