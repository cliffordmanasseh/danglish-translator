from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from danglish.backends.mock import MockBackend
from danglish.speech import SpeechTranslator


class FakeASR:
    def __init__(self, text: str = "Hej med dig. Hvordan går det?") -> None:
        self.text = text
        self.calls = 0
        self.closed = False

    def transcribe(self, audio: np.ndarray, *, sample_rate: int = 16_000) -> str:
        self.calls += 1
        return self.text

    def close(self) -> None:
        self.closed = True


class FakeTTS:
    sample_rate = 22_050

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.texts: list[str] = []
        self.closed = False

    def synthesize(self, text: str) -> np.ndarray:
        self.texts.append(text)
        if self.fail:
            raise RuntimeError("synthesis exploded")
        samples = int(self.sample_rate * 0.1)
        return np.full(samples, 0.05, dtype=np.float32)

    def close(self) -> None:
        self.closed = True


class RecordingBackend:
    name = "recording"

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def translate(self, texts: Sequence[str], *, beam_size: int | None = None) -> list[str]:
        self.batches.append(list(texts))
        return [f"EN({text})" for text in texts]

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_produces_text_and_audio_per_sentence() -> None:
    asr = FakeASR()
    tts = FakeTTS()
    backend = RecordingBackend()
    speech = SpeechTranslator(asr, backend, tts)

    result = speech.translate_utterance(np.zeros(1600, dtype=np.float32))

    assert result is not None
    assert result.transcript == "Hej med dig. Hvordan går det?"
    assert len(result.segments) == 2
    assert result.segments[0].target == "EN(Hej med dig.)"
    assert result.segments[1].target == "EN(Hvordan går det?)"
    assert all(segment.audio is not None for segment in result.segments)
    assert all(segment.sample_rate == 22_050 for segment in result.segments)
    assert tts.texts == ["EN(Hej med dig.)", "EN(Hvordan går det?)"]
    assert result.asr_seconds >= 0
    assert result.mt_seconds >= 0
    assert result.tts_seconds >= 0


def test_tts_failure_keeps_the_translation() -> None:
    speech = SpeechTranslator(FakeASR(), RecordingBackend(), FakeTTS(fail=True))

    result = speech.translate_utterance(np.zeros(1600, dtype=np.float32))

    assert result is not None
    assert result.segments
    assert result.segments[0].audio is None
    assert "RuntimeError" in (result.segments[0].tts_error or "")


def test_works_without_tts() -> None:
    speech = SpeechTranslator(FakeASR("Hej."), RecordingBackend(), None)

    result = speech.translate_utterance(np.zeros(1600, dtype=np.float32))

    assert result is not None
    assert result.segments[0].audio is None
    assert result.segments[0].tts_error is None


def test_empty_transcript_returns_none() -> None:
    asr = FakeASR("   ")
    tts = FakeTTS()
    backend = RecordingBackend()
    speech = SpeechTranslator(asr, backend, tts)

    assert speech.translate_utterance(np.zeros(1600, dtype=np.float32)) is None
    assert backend.batches == []
    assert tts.texts == []


def test_close_is_idempotent_and_closes_engines() -> None:
    asr = FakeASR()
    tts = FakeTTS()
    speech = SpeechTranslator(asr, MockBackend(), tts)

    speech.close()
    speech.close()

    assert asr.closed
    assert tts.closed
