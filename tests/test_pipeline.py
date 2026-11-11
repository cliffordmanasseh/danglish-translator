from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from danglish.backends.mock import MockBackend
from danglish.pipeline import StreamingPipeline, translate_document


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


def test_push_emits_final_event_with_id() -> None:
    pipeline = StreamingPipeline(MockBackend(), min_partial_chars=4)

    events = pipeline.push("Hej med dig. ")

    assert len(events) == 1
    event = events[0]
    assert event.kind == "final"
    assert event.id == 1
    assert event.source == "Hej med dig."
    assert event.target.startswith("[mock]")
    assert event.latency_ms is not None


def test_preview_translates_pending_text_and_deduplicates() -> None:
    pipeline = StreamingPipeline(MockBackend(), min_partial_chars=4)
    pipeline.push("Jeg vil gerne")

    first = pipeline.preview()
    assert first is not None
    assert first.kind == "partial"
    assert first.id is None
    assert first.source == "Jeg vil gerne"

    assert pipeline.preview() is None


def test_new_delta_invalidates_partial_preview() -> None:
    pipeline = StreamingPipeline(MockBackend(), min_partial_chars=4)
    pipeline.push("Jeg vil gerne")
    assert pipeline.preview() is not None

    pipeline.push(" bestille kaffe")

    assert pipeline.preview() is not None
    assert pipeline.pending.endswith("bestille kaffe")


def test_flush_commits_pending_text() -> None:
    pipeline = StreamingPipeline(MockBackend(), min_partial_chars=4)
    pipeline.push("En sætning uden punktum")

    events = pipeline.flush()

    assert len(events) == 1
    assert events[0].kind == "final"
    assert events[0].source == "En sætning uden punktum"
    assert pipeline.pending == ""


def test_reset_restarts_ids_and_clears_pending() -> None:
    pipeline = StreamingPipeline(MockBackend(), min_partial_chars=4)
    pipeline.push("Hej. ")
    pipeline.push("Noget mere")
    pipeline.reset()

    assert pipeline.pending == ""
    events = pipeline.push("Farvel. ")
    assert events[0].id == 1


def test_backend_is_called_once_for_a_batch_of_sentences() -> None:
    backend = RecordingBackend()
    pipeline = StreamingPipeline(backend, min_partial_chars=4)

    events = pipeline.push("En. To. Tre. ")

    assert len(events) == 3
    assert backend.batches == [["En.", "To.", "Tre."]]


def test_pending_buffer_is_bounded_without_punctuation() -> None:
    pipeline = StreamingPipeline(MockBackend(), min_partial_chars=2, max_pending_chars=20)
    text = "ord " * 30

    events = pipeline.push(text)

    assert events
    assert all(event.kind == "final" for event in events)
    assert len(pipeline.pending) <= 20


def test_preview_requires_minimum_length() -> None:
    pipeline = StreamingPipeline(MockBackend(), min_partial_chars=20)
    pipeline.push("kort")
    assert pipeline.preview() is None


def test_translate_document_preserves_paragraphs_and_batches() -> None:
    backend = RecordingBackend()
    text = "En. To. Tre.\n\nFire. Fem."

    result = translate_document(backend, text, batch_size=8)

    assert result == "EN(En.) EN(To.) EN(Tre.)\n\nEN(Fire.) EN(Fem.)"
    assert backend.batches == [["En.", "To.", "Tre."], ["Fire.", "Fem."]]


def test_translate_document_handles_blank_input() -> None:
    backend = RecordingBackend()
    assert translate_document(backend, "  \n\n ") == ""
    assert backend.batches == []


def test_translate_document_batches_long_paragraphs() -> None:
    backend = RecordingBackend()
    text = " ".join(f"Sætning {index}." for index in range(5))

    result = translate_document(backend, text, batch_size=2)

    assert result.startswith("EN(Sætning 0.)")
    assert [len(batch) for batch in backend.batches] == [2, 2, 1]


def test_event_to_dict_shape() -> None:
    pipeline = StreamingPipeline(MockBackend(), min_partial_chars=4)
    event = pipeline.push("Hej. ")[0]

    payload = event.to_dict()

    assert payload["type"] == "final"
    assert payload["id"] == 1
    assert "source" in payload and "target" in payload


def test_document_translation_rejects_empty_backend_result() -> None:
    class EmptyBackend(RecordingBackend):
        def translate(self, texts: Sequence[str], *, beam_size: int | None = None) -> list[str]:
            return ["" for _ in texts]

    result = translate_document(EmptyBackend(), "Hej.")
    assert result == ""


@pytest.mark.parametrize("paragraph", ["Hej.", "Hej", "Hej!\n\nFarvel."])
def test_document_never_loses_text(paragraph: str) -> None:
    backend = RecordingBackend()
    result = translate_document(backend, paragraph)
    assert result.strip()
    assert sum(len(batch) for batch in backend.batches) >= 1


def test_pathological_input_does_not_loop() -> None:
    pipeline = StreamingPipeline(MockBackend(), min_partial_chars=2, max_pending_chars=10)
    events = pipeline.push("x" * 100)
    assert events
    assert len(pipeline.pending) <= 10


def test_model_dir_type_is_not_required_for_document(tmp_path: Path) -> None:
    # A tiny guard for accidental required arguments in translate_document.
    backend = RecordingBackend()
    assert translate_document(backend, "Hej med dig.") == "EN(Hej med dig.)"
    assert not list(tmp_path.iterdir())
