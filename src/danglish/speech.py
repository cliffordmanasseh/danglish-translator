"""Speech-to-speech orchestration: ASR → translation → TTS.

The audio capture/playback loop lives in :mod:`danglish.audio.mic`; this module is
kept free of device dependencies so it can be tested with fake engines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .backends.base import TranslatorBackend
from .context import ContextProfile
from .pipeline import StreamingPipeline


class SpeechRecognizer(Protocol):
    def transcribe(self, audio: np.ndarray, *, sample_rate: int = 16_000) -> str: ...


class SpeechSynthesizer(Protocol):
    sample_rate: int

    def synthesize(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    source: str
    target: str
    audio: np.ndarray | None
    sample_rate: int | None
    tts_seconds: float = 0.0
    tts_error: str | None = None


@dataclass(frozen=True, slots=True)
class SpeechUtterance:
    transcript: str
    segments: tuple[SpeechSegment, ...]
    asr_seconds: float
    mt_seconds: float
    total_seconds: float

    @property
    def tts_seconds(self) -> float:
        return sum(segment.tts_seconds for segment in self.segments)


class SpeechTranslator:
    """Turn one Danish audio utterance into English text and speech."""

    def __init__(
        self,
        asr: SpeechRecognizer,
        backend: TranslatorBackend,
        tts: SpeechSynthesizer | None = None,
        *,
        min_partial_chars: int = 12,
        max_pending_chars: int = 4000,
        context: ContextProfile | None = None,
    ) -> None:
        self._asr = asr
        self._tts = tts
        self._pipeline = StreamingPipeline(
            backend,
            min_partial_chars=min_partial_chars,
            max_pending_chars=max_pending_chars,
            context=context,
        )
        self._closed = False

    def translate_utterance(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 16_000,
    ) -> SpeechUtterance | None:
        started = time.perf_counter()
        transcript = self._asr.transcribe(audio, sample_rate=sample_rate).strip()
        asr_seconds = time.perf_counter() - started
        if not transcript:
            return None

        mt_started = time.perf_counter()
        self._pipeline.reset()
        events = self._pipeline.push(transcript + " ")
        events.extend(self._pipeline.flush())
        mt_seconds = time.perf_counter() - mt_started

        segments: list[SpeechSegment] = []
        for event in events:
            if event.kind != "final":
                continue

            audio_out: np.ndarray | None = None
            sample_rate_out: int | None = None
            tts_seconds = 0.0
            tts_error: str | None = None

            if self._tts is not None:
                tts_started = time.perf_counter()
                try:
                    audio_out = self._tts.synthesize(event.target)
                    sample_rate_out = self._tts.sample_rate
                except Exception as exc:  # a TTS failure must not lose the translation
                    tts_error = f"{type(exc).__name__}: {exc}"
                tts_seconds = time.perf_counter() - tts_started

            segments.append(
                SpeechSegment(
                    source=event.source,
                    target=event.target,
                    audio=audio_out,
                    sample_rate=sample_rate_out,
                    tts_seconds=tts_seconds,
                    tts_error=tts_error,
                )
            )

        return SpeechUtterance(
            transcript=transcript,
            segments=tuple(segments),
            asr_seconds=asr_seconds,
            mt_seconds=mt_seconds,
            total_seconds=time.perf_counter() - started,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for engine in (self._asr, self._tts):
            close = getattr(engine, "close", None)
            if close is not None:
                close()
