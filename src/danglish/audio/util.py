"""Small audio helpers shared by ASR, TTS, and playback."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resampling; good enough for speech at these rates."""

    if source_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)

    duration = samples.size / float(source_rate)
    target_count = max(1, int(round(duration * target_rate)))
    source_times = np.linspace(0.0, duration, samples.size, endpoint=False)
    target_times = np.linspace(0.0, duration, target_count, endpoint=False)
    return np.interp(target_times, source_times, samples).astype(np.float32)


def write_wav(path: Path | str, audio: np.ndarray, sample_rate: int) -> Path:
    """Write mono float audio as a 16-bit PCM WAV file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return destination


def read_wav(path: Path | str) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit WAV file as float32 samples."""

    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError("expected a mono 16-bit PCM WAV file")
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sample_rate
