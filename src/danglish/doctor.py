"""Environment, dependency, and memory-budget diagnostics."""

from __future__ import annotations

import importlib.util
import os
import platform
import sys
from dataclasses import dataclass
from typing import Literal

from .config import AppConfig
from .memory import (
    DEFAULT_BUDGET_MB,
    estimate_memory,
    format_mb,
    memory_snapshot,
)
from .models import model_is_ready

Status = Literal["ok", "warn", "fail"]

_SYMBOLS: dict[Status, str] = {
    "ok": "✓",
    "warn": "!",
    "fail": "✗",
}


@dataclass(frozen=True, slots=True)
class Check:
    label: str
    status: Status
    detail: str


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _llama_health(server_url: str) -> tuple[bool, str]:
    import json
    import urllib.error
    import urllib.request

    url = f"{server_url.rstrip('/')}/health"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            data = json.loads(response.read())
        return True, f"llama.cpp server at {server_url} ({data.get('status', 'ok')})"
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        return False, (
            f"llama.cpp server at {server_url} not reachable ({exc}); "
            "start it from release/sermon-model/README.md"
        )


def build_checks(config: AppConfig, *, mock: bool = False) -> list[Check]:
    checks: list[Check] = []

    python_ok = sys.version_info >= (3, 10)
    checks.append(
        Check(
            "Python",
            "ok" if python_ok else "fail",
            f"{platform.python_version()} ({sys.executable})",
        )
    )
    checks.append(Check("Platform", "ok", f"{platform.system()} {platform.machine()}"))

    cpus = os.cpu_count() or 1
    checks.append(Check("CPU", "ok", f"{cpus} logical cores"))

    snapshot = memory_snapshot()
    if snapshot.total_mb is not None:
        checks.append(
            Check(
                "System RAM",
                "ok",
                f"{format_mb(snapshot.available_mb)} available of {format_mb(snapshot.total_mb)}",
            )
        )
    else:
        checks.append(Check("System RAM", "warn", "could not read /proc/meminfo"))

    if mock:
        checks.append(Check("Model", "warn", "mock backend selected; no model needed"))
        estimate = estimate_memory(None, with_server=True)
    elif config.model.backend == "llama":
        reachable, detail = _llama_health(config.model.server_url)
        checks.append(Check("Model", "ok" if reachable else "warn", detail))
        estimate = None
    else:
        ready = model_is_ready(config.model.path)
        if ready:
            estimate = estimate_memory(config.model.path, with_server=True)
            checks.append(
                Check(
                    "Model",
                    "ok",
                    f"{config.model.path} ({format_mb(estimate.weights_mb)} on disk, "
                    f"{config.model.compute_type})",
                )
            )
        else:
            estimate = estimate_memory(config.model.path, with_server=True)
            checks.append(
                Check(
                    "Model",
                    "fail",
                    f"not found at {config.model.path}; run `danglish fetch`",
                )
            )

    if config.model.backend == "llama":
        checks.append(
            Check(
                "Memory estimate",
                "ok",
                "~1.29 GB RSS for the Q5 sermon model (text only); "
                "the full speech stack plus this model exceeds the 2 GB budget",
            )
        )
        if snapshot.available_mb is not None and snapshot.available_mb < 1290:
            checks.append(
                Check(
                    "Available RAM",
                    "warn",
                    f"only {format_mb(snapshot.available_mb)} free; the Q5 model needs ~1.29 GB",
                )
            )
    else:
        budget_ok = estimate.fits(DEFAULT_BUDGET_MB)
        checks.append(
            Check(
                "Memory estimate",
                "ok" if budget_ok else "fail",
                f"~{format_mb(estimate.total_mb)} peak "
                f"(weights {format_mb(estimate.weights_mb)} + runtime) "
                f"against a {format_mb(DEFAULT_BUDGET_MB)} budget",
            )
        )
        if snapshot.available_mb is not None and snapshot.available_mb < estimate.total_mb:
            checks.append(
                Check(
                    "Available RAM",
                    "warn",
                    f"only {format_mb(snapshot.available_mb)} free; close other applications",
                )
            )

    runtime_modules = {
        "ctranslate2": "required",
        "sentencepiece": "required",
        "numpy": "required",
    }
    missing = [name for name in runtime_modules if not _has_module(name)]
    if missing:
        checks.append(Check("Runtime deps", "fail", "missing: " + ", ".join(missing)))
    else:
        checks.append(Check("Runtime deps", "ok", "ctranslate2 + sentencepiece + numpy"))

    optional = {
        "fastapi": "web UI",
        "uvicorn": "web UI",
        "faster_whisper": "speech input",
        "onnxruntime": "speech output",
        "dizione": "speech output",
        "piper": "Piper fallback",
        "sounddevice": "audio I/O",
    }
    installed = [label for module, label in optional.items() if _has_module(module)]
    checks.append(
        Check(
            "Optional extras",
            "ok",
            ", ".join(sorted(set(installed))) if installed else "none installed",
        )
    )

    if _has_module("faster_whisper") and _has_module("onnxruntime") and _has_module("dizione"):
        if config.model.backend == "llama":
            checks.append(
                Check(
                    "Speech stack",
                    "warn",
                    "ASR + TTS ~1.0 GB plus the Q5 model ~1.29 GB exceeds the 2 GB budget; "
                    "use text-only or raise the budget",
                )
            )
        else:
            checks.append(
                Check(
                    "Speech stack",
                    "ok",
                    "ASR small + translation + tiny-voca measured at ~1.0 GB peak; "
                    "fits the 2 GB budget",
                )
            )
    else:
        checks.append(
            Check(
                "Speech note",
                "warn",
                "audio-to-audio is optional; install with `uv sync --extra speech`",
            )
        )

    return checks


def overall_status(checks: list[Check]) -> Status:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "ok"


def render_checks(checks: list[Check]) -> str:
    width = max(len(check.label) for check in checks) if checks else 0
    lines = []
    for check in checks:
        lines.append(f"{_SYMBOLS[check.status]} {check.label:<{width}}  {check.detail}")
    return "\n".join(lines)
