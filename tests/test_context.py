from __future__ import annotations

from collections.abc import Sequence

import pytest

from danglish.backends.mock import MockBackend
from danglish.context import get_profile, list_profiles, profile_names
from danglish.errors import DanglishError
from danglish.pipeline import StreamingPipeline, translate_document


class FixedBackend:
    name = "fixed"

    def __init__(
        self, translations: dict[str, str] | None = None, default: str = "unchanged"
    ) -> None:
        self.translations = translations or {}
        self.default = default

    def translate(self, texts: Sequence[str], *, beam_size: int | None = None) -> list[str]:
        return [self.translations.get(text, self.default) for text in texts]

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_profile_names_and_lookup() -> None:
    assert "general" in profile_names()
    assert "sermon" in profile_names()
    assert get_profile(None).name == "general"
    assert get_profile("sermon").name == "sermon"
    assert len(list_profiles()) == 2

    with pytest.raises(DanglishError, match="unknown context profile"):
        get_profile("medical")


def test_sermon_terminology_preference() -> None:
    profile = get_profile("sermon")

    assert (
        profile.apply("Menigheden samledes søndag.", "The church gathered on Sunday.")
        == "The congregation gathered on Sunday."
    )
    assert (
        profile.apply("Præsten talte om nåde.", "The priest spoke about grace.")
        == "The pastor spoke about grace."
    )


def test_sermon_does_not_touch_unrelated_source() -> None:
    profile = get_profile("sermon")

    assert (
        profile.apply("Vi besøgte kirken i går.", "We visited the church yesterday.")
        == "We visited the church yesterday."
    )


def test_ambiguous_church_is_conservative() -> None:
    profile = get_profile("sermon")
    source = "Menigheden mødtes i kirken."
    target = "The church met in the church."

    # Both a congregation and a building are present; a blanket replacement
    # would corrupt the second "church".
    assert profile.apply(source, target) == target


def test_duplicate_pattern_is_not_replaced() -> None:
    profile = get_profile("sermon")
    source = "Menigheden voksede."
    target = "The church grew and the church was full."

    assert profile.apply(source, target) == target


def test_case_is_preserved() -> None:
    profile = get_profile("sermon")

    assert (
        profile.apply("Menigheden samledes.", "The Church gathered.")
        == "The Congregation gathered."
    )
    assert (
        profile.apply("Menigheden samledes.", "THE CHURCH GATHERED.")
        == "THE CONGREGATION GATHERED."
    )


def test_proper_noun_replacement_keeps_its_capitalization() -> None:
    profile = get_profile("sermon")

    assert (
        profile.apply("Helligånden kom over dem.", "the holy ghost came upon them.")
        == "the Holy Spirit came upon them."
    )


def test_general_profile_is_a_noop() -> None:
    profile = get_profile("general")
    target = "The church gathered."
    assert profile.apply("Menigheden samledes.", target) == target


def test_pipeline_applies_context_to_final_and_partial() -> None:
    backend = FixedBackend(
        {"Menigheden samledes.": "The church gathered."},
        default="The church is here.",
    )
    pipeline = StreamingPipeline(backend, min_partial_chars=4, context=get_profile("sermon"))

    final = pipeline.push("Menigheden samledes. ")[0]
    assert final.target == "The congregation gathered."

    pipeline.push("Menigheden")
    partial = pipeline.preview()
    assert partial is not None
    assert partial.target == "The congregation is here."


def test_document_applies_context() -> None:
    backend = FixedBackend(default="The church gathered.")

    result = translate_document(
        backend,
        "Menigheden samledes. Det var søndag.",
        context=get_profile("sermon"),
    )

    assert result.startswith("The congregation gathered.")


def test_context_still_works_with_real_mock_backend() -> None:
    pipeline = StreamingPipeline(MockBackend(), min_partial_chars=4, context=get_profile("sermon"))
    event = pipeline.push("Menigheden samledes. ")[0]
    assert event.target.startswith("[mock]")
