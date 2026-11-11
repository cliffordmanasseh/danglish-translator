"""Resident-memory measurement and model footprint estimation."""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:  # POSIX only; Danglish targets Linux and macOS for the memory tooling.
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

MB = 1024.0 * 1024.0
DEFAULT_BUDGET_MB = 2048

# Conservative constants for the interpreter, CT2, SentencePiece, NumPy, and
# model activations. Calibrated against `danglish bench` and the web server;
# measured values are ~147 MB (text) and ~185 MB (text + server). See docs.
_RUNTIME_MB = 70.0
_ACTIVATION_MB = 25.0
_SERVER_MB = 45.0


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    available_mb: float | None
    total_mb: float | None
    rss_mb: float
    peak_rss_mb: float


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    weights_mb: float
    runtime_mb: float
    activations_mb: float
    extras_mb: float

    @property
    def total_mb(self) -> float:
        return self.weights_mb + self.runtime_mb + self.activations_mb + self.extras_mb

    def headroom_mb(self, budget_mb: float = DEFAULT_BUDGET_MB) -> float:
        return budget_mb - self.total_mb

    def fits(self, budget_mb: float = DEFAULT_BUDGET_MB) -> bool:
        return self.total_mb <= budget_mb


def _read_meminfo() -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                number = rest.strip().split()
                if number:
                    with contextlib.suppress(ValueError):
                        values[key] = float(number[0]) / 1024.0  # kB -> MB
    except OSError:
        pass
    return values


def current_rss_mb() -> float:
    """Current resident set size of this process, in MiB."""

    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / MB
    except (OSError, IndexError, ValueError, AttributeError):
        if resource is not None:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak / MB if sys.platform == "darwin" else peak / 1024.0
        return 0.0


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB."""

    if resource is None:
        return current_rss_mb()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / MB if sys.platform == "darwin" else peak / 1024.0
    # /proc RSS can briefly report a few MB more than ru_maxrss; use the
    # conservative value so budget checks never under-report.
    return max(peak_mb, current_rss_mb())


def memory_snapshot() -> MemorySnapshot:
    meminfo = _read_meminfo()
    return MemorySnapshot(
        available_mb=meminfo.get("MemAvailable"),
        total_mb=meminfo.get("MemTotal"),
        rss_mb=current_rss_mb(),
        peak_rss_mb=peak_rss_mb(),
    )


def directory_size_mb(path: Path | str) -> float:
    root = Path(path)
    if not root.exists():
        return 0.0
    total = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
    return total / MB


def estimate_memory(
    model_dir: Path | str | None,
    *,
    with_server: bool = False,
) -> MemoryEstimate:
    weights = directory_size_mb(model_dir) if model_dir else 0.0
    return MemoryEstimate(
        weights_mb=weights,
        runtime_mb=_RUNTIME_MB,
        activations_mb=_ACTIVATION_MB,
        extras_mb=_SERVER_MB if with_server else 0.0,
    )


def format_mb(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "unknown"
    if signed:
        return f"{value:+,.0f} MB"
    return f"{value:,.0f} MB"
