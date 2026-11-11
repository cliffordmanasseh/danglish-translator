from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from danglish.audio.runtime import PlaybackGate, TalkRuntime
from danglish.speech import SpeechSegment, SpeechUtterance


class FakeVAD:
    def __init__(self) -> None:
        self.frames = 0
        self.resets = 0
        self._emitted = False

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        self.frames += 1
        if not self._emitted:
            self._emitted = True
            return [np.zeros(1600, dtype=np.float32)]
        return []

    def reset(self) -> None:
        self.resets += 1


class FakeSpeech:
    def __init__(self, error: bool = False) -> None:
        self.error = error
        self.calls = 0

    def translate_utterance(
        self, audio: np.ndarray, *, sample_rate: int = 16_000
    ) -> SpeechUtterance:
        self.calls += 1
        if self.error:
            raise RuntimeError("boom")
        return SpeechUtterance(
            transcript="Hej. Farvel.",
            segments=(
                SpeechSegment("Hej.", "Hello.", np.full(100, 0.1, dtype=np.float32), 16_000),
                SpeechSegment("Farvel.", "Goodbye.", np.full(100, 0.2, dtype=np.float32), 16_000),
            ),
            asr_seconds=0.01,
            mt_seconds=0.01,
            total_seconds=0.02,
        )


def test_half_duplex_gate_mutes_and_requests_reset() -> None:
    gate = PlaybackGate(full_duplex=False)
    assert gate.should_capture()

    gate.register(2)
    assert not gate.should_capture()

    gate.finished_one()
    assert not gate.should_capture()
    assert not gate.consume_reset()

    gate.finished_one()
    assert gate.should_capture()
    assert gate.consume_reset()
    assert not gate.consume_reset()


def test_full_duplex_gate_never_mutes() -> None:
    gate = PlaybackGate(full_duplex=True)
    gate.register(1)

    assert gate.should_capture()
    assert gate.pending == 1

    gate.finished_one()
    assert gate.should_capture()
    assert not gate.consume_reset()


def test_capture_is_dropped_while_half_duplex_gate_is_closed() -> None:
    runtime = TalkRuntime(FakeSpeech(), lambda audio, rate: None, FakeVAD())
    runtime.gate.register(1)

    runtime.capture(np.zeros(480, dtype=np.float32))
    assert runtime.dropped_frames == 1

    runtime.gate.finished_one()
    runtime.capture(np.zeros(480, dtype=np.float32))
    assert runtime.dropped_frames == 1


def test_capture_continues_during_playback_in_full_duplex() -> None:
    runtime = TalkRuntime(
        FakeSpeech(),
        lambda audio, rate: None,
        FakeVAD(),
        full_duplex=True,
    )
    runtime.gate.register(1)

    runtime.capture(np.zeros(480, dtype=np.float32))

    assert runtime.dropped_frames == 0


def test_runtime_translates_and_plays_in_order() -> None:
    emitted: list[SpeechUtterance] = []
    played: list[float] = []
    done = threading.Event()

    def play(audio: np.ndarray, sample_rate: int) -> None:
        played.append(float(audio[0]))
        if len(played) == 2:
            done.set()

    runtime = TalkRuntime(
        FakeSpeech(),
        play,
        FakeVAD(),
        full_duplex=True,
        on_utterance=emitted.append,
    )
    runtime.start()
    vad_thread = threading.Thread(target=runtime.run_vad_loop, daemon=True)
    vad_thread.start()
    try:
        runtime.capture(np.zeros(480, dtype=np.float32))
        assert done.wait(3.0)
    finally:
        runtime.stop()
        runtime.join()
        vad_thread.join(3.0)

    assert played == [pytest.approx(0.1, rel=1e-4), pytest.approx(0.2, rel=1e-4)]
    assert len(emitted) == 1
    assert [segment.target for segment in emitted[0].segments] == ["Hello.", "Goodbye."]
    assert runtime.dropped_frames == 0


def test_runtime_survives_a_translation_error() -> None:
    errors: list[str] = []
    error_seen = threading.Event()

    def on_error(message: str) -> None:
        errors.append(message)
        error_seen.set()

    runtime = TalkRuntime(
        FakeSpeech(error=True),
        lambda audio, rate: None,
        FakeVAD(),
        full_duplex=True,
        on_error=on_error,
    )
    runtime.start()
    vad_thread = threading.Thread(target=runtime.run_vad_loop, daemon=True)
    vad_thread.start()
    try:
        runtime.capture(np.zeros(480, dtype=np.float32))
        assert error_seen.wait(3.0)
    finally:
        runtime.stop()
        runtime.join()
        vad_thread.join(3.0)

    assert any("translation failed" in message for message in errors)


def test_stop_is_idempotent() -> None:
    runtime = TalkRuntime(FakeSpeech(), lambda audio, rate: None, FakeVAD())
    runtime.stop()
    runtime.stop()
    runtime.join(timeout=0.1)
    time.sleep(0.01)
