from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from danglish.audio.util import read_wav, resample_audio, write_wav


def test_resample_length() -> None:
    source = np.sin(np.linspace(0, 2 * np.pi, 16_000, endpoint=False)).astype(np.float32)
    resampled = resample_audio(source, 16_000, 22_050)
    assert abs(resampled.size - 22_050) <= 1
    assert resampled.dtype == np.float32


def test_resample_noop() -> None:
    source = np.ones(100, dtype=np.float32)
    assert resample_audio(source, 16_000, 16_000) is source


def test_wav_roundtrip(tmp_path: Path) -> None:
    audio = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    path = write_wav(tmp_path / "tone.wav", audio, 22_050)
    restored, rate = read_wav(path)

    assert rate == 22_050
    assert restored.size == audio.size
    assert np.max(np.abs(restored - audio)) < 1e-3


def test_wav_clips_out_of_range(tmp_path: Path) -> None:
    audio = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    path = write_wav(tmp_path / "clipped.wav", audio, 16_000)
    restored, _ = read_wav(path)
    assert np.max(restored) <= 1.0
    assert np.min(restored) >= -1.0


def test_read_wav_rejects_stereo(tmp_path: Path) -> None:
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0" * 8)

    with pytest.raises(ValueError, match="mono"):
        read_wav(path)
