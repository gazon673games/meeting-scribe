# -*- mode: python ; coding: utf-8 -*-

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


ROOT = Path(SPECPATH).resolve().parents[1]
icon_path = ROOT / "frontend" / "electron" / "assets" / "icon.ico"
datas = []
binaries = []
hiddenimports = []

config_path = ROOT / "config.json"
if config_path.exists():
    datas.append((str(config_path), "."))

for package_name in (
    "av",
    "ctranslate2",
    "faster_whisper",
    "onnxruntime",
    "soundcard",
    "sounddevice",
    "soundfile",
    "tokenizers",
):
    if importlib.util.find_spec(package_name) is None:
        continue
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports


a = Analysis(
    [str(ROOT / "backend" / "main_electron_backend.py")],
    pathex=[str(ROOT / "backend"), str(ROOT / "backend" / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "nemo",
        "nemo_toolkit",
        "pyannote",
        "pyannote.audio",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="meeting-scribe-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path) if icon_path.exists() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="meeting-scribe-backend",
)
