from __future__ import annotations

from pathlib import Path

from danglish.models import model_is_ready


def test_model_is_ready_requires_all_files(tmp_path: Path) -> None:
    for name in ("model.bin", "source.spm", "target.spm"):
        (tmp_path / name).write_bytes(b"x")
    assert model_is_ready(tmp_path)

    (tmp_path / "target.spm").unlink()
    assert not model_is_ready(tmp_path)


def test_model_is_ready_missing_directory(tmp_path: Path) -> None:
    assert not model_is_ready(tmp_path / "missing")
