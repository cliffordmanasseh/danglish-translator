"""Latency and memory benchmark for a translation backend."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .backends.base import TranslatorBackend
from .corpus import BENCHMARK_SENTENCES
from .memory import (
    DEFAULT_BUDGET_MB,
    current_rss_mb,
    directory_size_mb,
    peak_rss_mb,
)


@dataclass(frozen=True, slots=True)
class BenchSample:
    source: str
    target: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class BenchResult:
    sentences: int
    source_words: int
    target_words: int
    total_seconds: float
    p50_ms: float
    p95_ms: float
    mean_ms: float
    target_words_per_second: float
    rss_after_load_mb: float
    peak_rss_mb: float
    weights_mb: float
    budget_mb: int
    within_budget: bool
    samples: tuple[BenchSample, ...] = field(default_factory=tuple)

    def to_dict(self, *, include_samples: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "sentences": self.sentences,
            "source_words": self.source_words,
            "target_words": self.target_words,
            "total_seconds": round(self.total_seconds, 4),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "mean_ms": round(self.mean_ms, 2),
            "target_words_per_second": round(self.target_words_per_second, 1),
            "rss_after_load_mb": round(self.rss_after_load_mb, 1),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "weights_mb": round(self.weights_mb, 1),
            "budget_mb": self.budget_mb,
            "within_budget": self.within_budget,
        }
        if include_samples:
            payload["samples"] = [
                {
                    "source": sample.source,
                    "target": sample.target,
                    "latency_ms": round(sample.latency_ms, 2),
                }
                for sample in self.samples
            ]
        return payload


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * fraction)))
    return sorted_values[index]


def run_benchmark(
    backend: TranslatorBackend,
    *,
    model_dir: Path | str | None = None,
    sentences: tuple[str, ...] = BENCHMARK_SENTENCES,
    repeats: int = 1,
    budget_mb: int = DEFAULT_BUDGET_MB,
) -> BenchResult:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    backend.warmup()
    rss_after_load = current_rss_mb()

    samples: list[BenchSample] = []
    durations: list[float] = []
    started_all = time.perf_counter()

    for _ in range(repeats):
        for sentence in sentences:
            started = time.perf_counter()
            target = backend.translate([sentence])[0]
            duration = (time.perf_counter() - started) * 1000.0
            durations.append(duration)
            samples.append(BenchSample(sentence, target, duration))

    total_seconds = time.perf_counter() - started_all
    durations.sort()
    source_words = sum(len(sentence.split()) for sentence in sentences) * repeats
    target_words = sum(len(sample.target.split()) for sample in samples)

    return BenchResult(
        sentences=len(samples),
        source_words=source_words,
        target_words=target_words,
        total_seconds=total_seconds,
        p50_ms=_percentile(durations, 0.50),
        p95_ms=_percentile(durations, 0.95),
        mean_ms=statistics.fmean(durations) if durations else 0.0,
        target_words_per_second=target_words / total_seconds if total_seconds else 0.0,
        rss_after_load_mb=rss_after_load,
        peak_rss_mb=peak_rss_mb(),
        weights_mb=directory_size_mb(model_dir) if model_dir else 0.0,
        budget_mb=budget_mb,
        within_budget=peak_rss_mb() <= budget_mb,
        samples=tuple(samples),
    )
