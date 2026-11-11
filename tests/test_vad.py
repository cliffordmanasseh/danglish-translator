from __future__ import annotations

import numpy as np

from danglish.audio.vad import EnergyVAD

SAMPLE_RATE = 16_000
FRAME = 480


def _speech(frames: int = 1) -> list[np.ndarray]:
    t = np.arange(FRAME * frames, dtype=np.float32) / SAMPLE_RATE
    wave = 0.08 * np.sin(2 * np.pi * 220 * t)
    return [wave[index : index + FRAME] for index in range(0, wave.size, FRAME)]


def _silence(frames: int = 1) -> list[np.ndarray]:
    return [np.zeros(FRAME, dtype=np.float32) for _ in range(frames)]


def test_detects_one_utterance_and_trims_silence() -> None:
    vad = EnergyVAD(end_silence_ms=300, min_utterance_s=0.2)
    completed: list[np.ndarray] = []
    for frame in _speech(40) + _silence(20):
        completed.extend(vad.push(frame))

    assert len(completed) == 1
    duration = len(completed[0]) / SAMPLE_RATE
    assert 0.9 <= duration <= 1.8


def test_silence_only_produces_nothing() -> None:
    vad = EnergyVAD()
    for frame in _silence(100):
        assert vad.push(frame) == []
    assert not vad.active
    assert vad.flush() == []


def test_flush_returns_active_utterance() -> None:
    vad = EnergyVAD(end_silence_ms=5000, min_utterance_s=0.2)
    for frame in _speech(30):
        vad.push(frame)
    assert vad.active

    completed = vad.flush()
    assert len(completed) == 1
    assert vad.flush() == []
    assert not vad.active


def test_short_blip_is_rejected() -> None:
    vad = EnergyVAD(end_silence_ms=300, min_utterance_s=0.4)
    completed: list[np.ndarray] = []
    for frame in _speech(5) + _silence(20):
        completed.extend(vad.push(frame))
    assert completed == []


def test_two_utterances() -> None:
    vad = EnergyVAD(end_silence_ms=300, min_utterance_s=0.2)
    completed: list[np.ndarray] = []
    for frame in _speech(30) + _silence(15) + _speech(30) + _silence(15):
        completed.extend(vad.push(frame))
    assert len(completed) == 2


def test_stereo_input_is_downmixed() -> None:
    vad = EnergyVAD()
    stereo = np.zeros((FRAME, 2), dtype=np.float32)
    assert vad.push(stereo) == []


def test_reset_drops_in_progress_audio() -> None:
    vad = EnergyVAD(end_silence_ms=5000, min_utterance_s=0.2)
    for frame in _speech(30):
        vad.push(frame)
    assert vad.active

    vad.reset()

    assert not vad.active
    assert vad.flush() == []
    assert vad.push(_silence(1)[0]) == []
