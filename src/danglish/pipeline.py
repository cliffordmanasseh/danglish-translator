"""Streaming translation pipeline: segment, batch, translate, preview."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Literal

from .backends.base import TranslatorBackend
from .context import ContextProfile
from .errors import BackendError
from .segmenter import SentenceSegmenter


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One translation event produced by the pipeline."""

    kind: Literal["final", "partial"]
    source: str
    target: str
    id: int | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "type": self.kind,
            "source": self.source,
            "target": self.target,
        }
        if self.id is not None:
            payload["id"] = self.id
        if self.latency_ms is not None:
            payload["latency_ms"] = round(self.latency_ms, 2)
        return payload


class StreamingPipeline:
    """Translate an incrementally growing Danish text stream.

    ``push()`` emits final events for completed sentences. ``preview()`` translates
    the unfinished tail; callers should debounce it. ``flush()`` commits whatever is
    left as a final event.
    """

    def __init__(
        self,
        backend: TranslatorBackend,
        *,
        min_partial_chars: int = 12,
        max_pending_chars: int = 4000,
        context: ContextProfile | None = None,
    ) -> None:
        self._backend = backend
        self._context = context
        self._segmenter = SentenceSegmenter()
        self._min_partial_chars = min_partial_chars
        self._max_pending_chars = max_pending_chars
        self._next_id = 1
        self._last_partial: str | None = None

    @property
    def pending(self) -> str:
        return self._segmenter.pending

    @property
    def backend(self) -> TranslatorBackend:
        return self._backend

    def reset(self) -> None:
        self._segmenter.reset()
        self._next_id = 1
        self._last_partial = None

    def push(self, delta: str) -> list[StreamEvent]:
        """Append text, returning final translations for completed sentences."""

        if not delta:
            return []

        self._last_partial = None
        events = self._translate_finals(self._segmenter.feed(delta))

        # Bound the buffer for streams that never contain sentence punctuation
        # (some ASR outputs). Prefer a clause boundary, otherwise a hard cut.
        while len(self._segmenter.pending) > self._max_pending_chars:
            piece = self._segmenter.take(self._max_pending_chars)
            if not piece:
                break
            events.extend(self._translate_finals([piece]))

        return events

    def preview(self) -> StreamEvent | None:
        """Translate the pending fragment, if it changed enough to be worth it."""

        source = self._segmenter.pending
        if len(source) < self._min_partial_chars:
            return None
        if source == self._last_partial:
            return None

        started = time.perf_counter()
        target = self._backend.translate([source])[0]
        if self._context is not None:
            target = self._context.apply(source, target)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._last_partial = source
        return StreamEvent(
            "partial",
            source=source,
            target=target,
            latency_ms=round(latency_ms, 2),
        )

    def flush(self) -> list[StreamEvent]:
        """Finish the current utterance and translate any remaining text."""

        self._last_partial = None
        return self._translate_finals(self._segmenter.flush())

    def _translate_finals(self, segments: list[str]) -> list[StreamEvent]:
        if not segments:
            return []

        started = time.perf_counter()
        targets = self._backend.translate(segments)
        latency_ms = (time.perf_counter() - started) * 1000.0

        if len(targets) != len(segments):
            raise BackendError(
                f"backend returned {len(targets)} translations for {len(segments)} inputs"
            )

        events: list[StreamEvent] = []
        for source, target in zip(segments, targets, strict=True):
            if self._context is not None:
                target = self._context.apply(source, target)
            events.append(
                StreamEvent(
                    "final",
                    source=source,
                    target=target,
                    id=self._next_id,
                    latency_ms=round(latency_ms, 2),
                )
            )
            self._next_id += 1
        return events


def translate_document(
    backend: TranslatorBackend,
    text: str,
    *,
    batch_size: int = 8,
    context: ContextProfile | None = None,
) -> str:
    """Translate a whole document while preserving paragraph breaks.

    Sentences within each paragraph are batched for throughput. This path is used by
    the one-shot CLI; the streaming pipeline is used for real-time input.
    """

    if not text.strip():
        return ""

    paragraphs = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    translated_paragraphs: list[str] = []

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        segmenter = SentenceSegmenter()
        normalized = " ".join(paragraph.split()) + " "
        sentences = segmenter.feed(normalized)
        sentences.extend(segmenter.flush())
        if not sentences:
            continue

        translated: list[str] = []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            targets = backend.translate(batch)
            for source, target in zip(batch, targets, strict=True):
                translated.append(context.apply(source, target) if context is not None else target)

        text_out = " ".join(part for part in translated if part).strip()
        if text_out:
            translated_paragraphs.append(text_out)

    return "\n\n".join(translated_paragraphs)
