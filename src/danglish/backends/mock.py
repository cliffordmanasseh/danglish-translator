"""Deterministic fake backend for tests and demos without a model."""

from __future__ import annotations

from collections.abc import Sequence


class MockBackend:
    """Pseudo-translates by reversing word order.

    It is intentionally obvious that the output is fake. Useful for UI work,
    tests, and checking the streaming behaviour on a machine without a model.
    """

    name = "mock"

    def __init__(self, marker: str = "[mock]") -> None:
        self.marker = marker

    def translate(self, texts: Sequence[str], *, beam_size: int | None = None) -> list[str]:
        return [self._pseudo(text) for text in texts]

    def warmup(self) -> None:
        return None

    def close(self) -> None:
        return None

    def _pseudo(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        words = stripped.split()
        reversed_words = " ".join(reversed(words))
        return f"{self.marker} {reversed_words}"
