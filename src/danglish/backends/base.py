"""Translation backend contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class TranslatorBackend(Protocol):
    """Minimal interface implemented by translation backends."""

    name: str

    def translate(self, texts: Sequence[str], *, beam_size: int | None = None) -> list[str]:
        """Translate a batch of Danish texts, preserving order."""

    def warmup(self) -> None:
        """Run a tiny translation so the first real request is fast."""

    def close(self) -> None:
        """Release model resources."""
