from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def per_process_audio_supported() -> bool:
    try:
        from infrastructure.process_session_catalog import is_per_process_audio_supported

        return is_per_process_audio_supported()
    except Exception:
        return False


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def safe_token_preview(token: object) -> str:
    try:
        text = repr(token)
    except Exception:
        return type(token).__name__
    if len(text) > 180:
        return f"{text[:177]}..."
    return text


def cpu_name() -> str:
    if platform.system().lower() == "windows":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                text = str(value).strip()
                if text:
                    return text
        except Exception:
            pass
    return platform.processor() or platform.machine() or "CPU"


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def physical_memory_bytes() -> int:
    return memory_snapshot().get("totalMemoryBytes", 0)


def memory_snapshot() -> Dict[str, int]:
    if platform.system().lower() == "windows":
        try:
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return {
                    "totalMemoryBytes": int(status.ullTotalPhys),
                    "freeMemoryBytes": int(status.ullAvailPhys),
                }
        except Exception:
            return {"totalMemoryBytes": 0, "freeMemoryBytes": 0}
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return {
            "totalMemoryBytes": int(pages) * int(page_size),
            "freeMemoryBytes": int(available_pages) * int(page_size),
        }
    except Exception:
        return {"totalMemoryBytes": 0, "freeMemoryBytes": 0}


def current_process_memory_bytes() -> int:
    if platform.system().lower() == "windows":
        try:
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
            kernel32 = ctypes.WinDLL("Kernel32.dll", use_last_error=True)
            psapi = ctypes.WinDLL("Psapi.dll", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_ProcessMemoryCounters),
                ctypes.c_ulong,
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_bool
            handle = kernel32.GetCurrentProcess()
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return int(counters.WorkingSetSize)
        except Exception:
            return 0
    try:
        statm = Path("/proc/self/statm")
        if statm.exists():
            pages = int(statm.read_text(encoding="utf-8").split()[1])
            return pages * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        pass
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system().lower() == "darwin" else value * 1024
    except Exception:
        return 0


def nvidia_gpu_snapshot() -> List[Dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    command = [
        executable,
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True, timeout=2.0)
    except Exception:
        return []
    if completed.returncode != 0:
        return []

    gpus: List[Dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        memory_total_mib = int_or_zero(parts[1])
        memory_used_mib = int_or_zero(parts[2])
        gpus.append(
            {
                "name": parts[0],
                "memoryTotalMiB": memory_total_mib,
                "memoryUsedMiB": memory_used_mib,
                "memoryFreeMiB": max(0, memory_total_mib - memory_used_mib),
                "gpuUtilizationPct": int_or_zero(parts[3]),
                "memoryUtilizationPct": int_or_zero(parts[4]),
            }
        )
    return gpus


def scan_compatible_asr_models(models_dir: Any) -> List[str]:
    try:
        from application.model_download import scan_local_models

        return [
            str(m.get("name") or "")
            for m in scan_local_models(models_dir)
            if m.get("compatible") and str(m.get("name") or "").strip()
        ]
    except Exception:
        return []


def int_or_zero(raw: object) -> int:
    return int_or_default(raw, 0)


def int_or_default(raw: object, default: int) -> int:
    try:
        return int(float(str(raw).strip()))
    except Exception:
        return int(default)

