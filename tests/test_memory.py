from __future__ import annotations

from pathlib import Path

import pytest

from danglish.memory import (
    current_rss_mb,
    directory_size_mb,
    estimate_memory,
    format_mb,
    memory_snapshot,
    peak_rss_mb,
)


def test_estimate_counts_weights_and_runtime(tmp_path: Path) -> None:
    (tmp_path / "model.bin").write_bytes(b"\0" * (1024 * 1024))
    (tmp_path / "source.spm").write_bytes(b"\0" * 1024)

    estimate = estimate_memory(tmp_path)

    assert estimate.weights_mb == pytest.approx(1.0009765625)
    assert estimate.total_mb == pytest.approx(1.0009765625 + 70 + 25)
    assert estimate.fits(2048)
    assert estimate.headroom_mb(2048) > 1900


def test_estimate_with_server_extra(tmp_path: Path) -> None:
    plain = estimate_memory(tmp_path)
    served = estimate_memory(tmp_path, with_server=True)
    assert served.total_mb == plain.total_mb + 45


def test_missing_model_directory_is_zero(tmp_path: Path) -> None:
    assert directory_size_mb(tmp_path / "missing") == 0.0
    assert estimate_memory(None).weights_mb == 0.0


def test_format_mb() -> None:
    assert format_mb(None) == "unknown"
    assert format_mb(1024) == "1,024 MB"
    assert format_mb(-5, signed=True) == "-5 MB"


def test_current_rss_is_positive() -> None:
    assert current_rss_mb() > 0
    assert peak_rss_mb() >= current_rss_mb() * 0.5
    snapshot = memory_snapshot()
    assert snapshot.rss_mb > 0
    assert snapshot.peak_rss_mb >= snapshot.rss_mb * 0.5


def test_format_is_compact_for_doctor() -> None:
    assert "MB" in format_mb(250)
