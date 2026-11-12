# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for the Danglish desktop backend.

Build with:
    python -m PyInstaller packaging/danglish-backend.spec \
        --noconfirm --clean \
        --distpath desktop/resources/backend \
        --workpath build/pyinstaller

The model files are not part of the sidecar; Electron ships them as resources.
"""

import subprocess
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

root = Path(SPECPATH).parent

datas = []
binaries = []
hiddenimports = []

# Native runtimes and package data. collect_all is deliberate: these packages use
# dynamic imports and data files that PyInstaller's static analysis misses.
for package in (
    "ctranslate2",
    "sentencepiece",
    "faster_whisper",
    "kokoro_onnx",
    "phonemizer",
    "espeakng_loader",
    "piper",
    "onnxruntime",
    "sounddevice",
    "av",
    "tokenizers",
    "huggingface_hub",
    "pathvalidate",
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

hiddenimports += [
    "danglish.audio.asr",
    "danglish.audio.check",
    "danglish.audio.device",
    "danglish.audio.mic",
    "danglish.audio.runtime",
    "danglish.context",
    "danglish.speech",
    "danglish.tts",
    "danglish.audio.util",
    "danglish.audio.vad",
]

# sounddevice loads PortAudio through ctypes. On Linux, copy the system library so
# the AppImage does not depend on the host having libportaudio installed.
if sys.platform.startswith("linux"):
    try:
        output = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, check=False
        ).stdout
        for line in output.splitlines():
            if "libportaudio.so.2" in line and "=>" in line:
                candidate = line.split("=>")[-1].strip()
                if Path(candidate).is_file():
                    binaries.append((candidate, "."))
                    break
    except (OSError, subprocess.SubprocessError):
        pass

a = Analysis(
    [str(root / "packaging" / "backend_entry.py")],
    pathex=[str(root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "torch",
        "transformers",
        "fastapi",
        "uvicorn",
        "starlette",
        "pydantic",
        "httpx",
        "matplotlib",
        "pytest",
        "tkinter",
        "PyQt5",
        "PyQt6",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="danglish-backend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="danglish-backend",
)
