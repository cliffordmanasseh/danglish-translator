from __future__ import annotations

import os
from pathlib import Path

import pytest

from danglish.backends.ct2 import CT2Backend
from danglish.errors import ModelNotFoundError
from danglish.models import model_is_ready
from danglish.pipeline import translate_document

MODEL_DIR = Path(os.environ.get("DANGLISH_MODEL", "models/opus-mt-da-en-int8"))

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def backend() -> CT2Backend:
    if not model_is_ready(MODEL_DIR):
        pytest.skip(f"no converted model at {MODEL_DIR}; run `danglish fetch`")
    instance = CT2Backend(MODEL_DIR, compute_type="int8", beam_size=2)
    yield instance
    instance.close()


def test_translates_a_basic_sentence(backend: CT2Backend) -> None:
    result = backend.translate(["God morgen, hvordan har du det i dag?"])[0]

    assert result == "Good morning, how are you today?"


def test_no_repetition_loop(backend: CT2Backend) -> None:
    # Regression test for the missing source EOS token, which made this model
    # repeat "morning morning morning ..." until the decoding limit.
    result = backend.translate(["God morgen, hvordan har du det i dag?"])[0]

    assert result.lower().count("morning") == 1
    assert len(result.split()) <= 12
    assert "</s>" not in result


def test_batch_preserves_order(backend: CT2Backend) -> None:
    first, second = backend.translate(["Jeg elsker kaffe.", "Han bor i Odense."])

    assert "coffee" in first.lower()
    assert "Odense" in second
    assert first != second


def test_empty_input(backend: CT2Backend) -> None:
    assert backend.translate([""]) == [""]
    assert backend.translate([]) == []


def test_document_translation(backend: CT2Backend) -> None:
    result = translate_document(
        backend,
        "Kommunen har besluttet at renovere den gamle skole. "
        "Det bliver færdigt næste år.\n\n"
        "Vi glæder os.",
    )

    assert "municipality" in result.lower()
    assert "\n\n" in result
    assert "looking forward" in result.lower() or "glad" in result.lower()


def test_missing_model_is_a_clean_error(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="danglish fetch"):
        CT2Backend(tmp_path)


@pytest.mark.skipif(not model_is_ready(MODEL_DIR), reason="model not available")
def test_model_directory_is_ready() -> None:
    assert model_is_ready(MODEL_DIR)
