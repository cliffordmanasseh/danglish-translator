from __future__ import annotations

import pytest

from danglish.segmenter import SentenceSegmenter


def test_basic_sentences_are_emitted_as_soon_as_they_are_complete() -> None:
    segmenter = SentenceSegmenter()

    assert segmenter.feed("Hej med dig. ") == ["Hej med dig."]
    assert segmenter.feed("Hvordan går det? ") == ["Hvordan går det?"]
    assert segmenter.pending == ""
    assert segmenter.flush() == []


def test_abbreviations_do_not_split() -> None:
    segmenter = SentenceSegmenter()
    text = "Jeg bor i København, bl.a. på Nørrebro. Det er dejligt. "

    assert segmenter.feed(text) == [
        "Jeg bor i København, bl.a. på Nørrebro.",
        "Det er dejligt.",
    ]


def test_thousands_and_times_do_not_split() -> None:
    segmenter = SentenceSegmenter()
    text = "Det kostede 1.000 kroner, og toget går kl. 14.30. Vi ses i morgen. "

    assert segmenter.feed(text) == [
        "Det kostede 1.000 kroner, og toget går kl. 14.30.",
        "Vi ses i morgen.",
    ]


def test_initials_do_not_split() -> None:
    segmenter = SentenceSegmenter()
    text = "J. K. Rowling skrev bogen. Den er god. "

    assert segmenter.feed(text) == ["J. K. Rowling skrev bogen.", "Den er god."]


def test_period_inside_quoted_sentence_does_not_split() -> None:
    segmenter = SentenceSegmenter()
    text = '"Hej." sagde hun. Hun gik videre. '

    assert segmenter.feed(text) == ['"Hej." sagde hun.', "Hun gik videre."]


def test_ellipsis_is_one_boundary() -> None:
    segmenter = SentenceSegmenter()
    text = "Jeg ved ikke... Måske kommer han. "

    assert segmenter.feed(text) == ["Jeg ved ikke...", "Måske kommer han."]


def test_incremental_feeding_matches_bulk_feeding() -> None:
    text = "Hej! Hvordan går det? Det er fint. "
    incremental: list[str] = []
    segmenter = SentenceSegmenter()
    for character in text:
        incremental.extend(segmenter.feed(character))

    bulk = SentenceSegmenter()
    assert incremental == bulk.feed(text)
    assert incremental == ["Hej!", "Hvordan går det?", "Det er fint."]


def test_final_fragment_requires_flush() -> None:
    segmenter = SentenceSegmenter()
    assert segmenter.feed("Dette er en sætning uden punktum") == []
    assert segmenter.pending == "Dette er en sætning uden punktum"
    assert segmenter.flush() == ["Dette er en sætning uden punktum"]
    assert segmenter.pending == ""


def test_lowercase_continuation_does_not_split() -> None:
    segmenter = SentenceSegmenter()
    # A conservative heuristic: a stop followed by a lower-case word is treated
    # as an abbreviation/continuation rather than a sentence boundary.
    assert segmenter.feed("Det er f.eks. meget svært. ") == ["Det er f.eks. meget svært."]


def test_take_bounds_the_pending_buffer() -> None:
    segmenter = SentenceSegmenter()
    segmenter.feed("a" * 50)

    piece = segmenter.take(20)
    assert piece == "a" * 20
    assert len(segmenter.pending) == 30


def test_empty_and_whitespace_input() -> None:
    segmenter = SentenceSegmenter()
    assert segmenter.feed("") == []
    assert segmenter.feed("   ") == []
    assert segmenter.pending == ""
    assert segmenter.flush() == []


@pytest.mark.parametrize(
    "text",
    [
        'Hun sagde: "Kom nu." Så gik vi.',
        "Er det sandt?! Ja, det er det.",
        "Prisen er 1.234,56 kr. Det er billigt.",
        "Vi mødes den 1. maj kl. 10.00.",
    ],
)
def test_segmenter_always_makes_progress(text: str) -> None:
    segmenter = SentenceSegmenter()
    segments = segmenter.feed(text)
    segments.extend(segmenter.flush())
    assert segments
    assert all(segment.strip() for segment in segments)
