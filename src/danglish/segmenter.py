"""Incremental Danish sentence segmentation.

The segmenter receives text deltas and emits complete sentences. It is intentionally
conservative: it prefers waiting for more text over cutting a sentence in the wrong
place, because a wrong cut produces a visibly wrong translation.

Known limitations (documented in ``docs/architecture.md``):

* an abbreviation that genuinely ends a sentence may delay the cut until ``flush()``;
* lists or headings without terminal punctuation are emitted only on flush.
"""

from __future__ import annotations

import re
from typing import Final

_TERMINATORS: Final = ".!?…"
_CLOSERS: Final = "\"'”’»«"
_OPENERS: Final = "([{"

# Lower-cased, without the trailing period. Entries may contain internal periods.
_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "bl.a",
        "ca",
        "d",
        "dvs",
        "d.v.s",
        "e.lign",
        "etc",
        "evt",
        "f.eks",
        "f.om",
        "fhv",
        "fx",
        "hr",
        "ift",
        "jf",
        "jvf",
        "kl",
        "m.fl",
        "m.m",
        "mht",
        "nr",
        "o.a",
        "o.lign",
        "osv",
        "pga",
        "pkt",
        "pr",
        "prof",
        "stk",
        "skt",
        "tlf",
        "u.p.t",
        "vedr",
        "vha",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "okt",
        "nov",
        "dec",
    }
)

# The immediately preceding word. Dots are included so "f.eks" is one token.
_PRECEDING_TOKEN = re.compile(r"([0-9A-Za-zÆØÅæøå.]+)$")


def _normalize(text: str) -> str:
    return " ".join(text.split())


class SentenceSegmenter:
    """Incremental sentence splitter for Danish text."""

    def __init__(self) -> None:
        self._buffer = ""

    @property
    def pending(self) -> str:
        """Text received but not yet emitted as a complete sentence."""

        return self._buffer.strip()

    def reset(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        """Append text and return any sentences completed by it."""

        if not delta:
            return []
        self._buffer += delta
        return self._pop(final=False)

    def flush(self) -> list[str]:
        """Emit everything that is still buffered, including a final fragment."""

        segments = self._pop(final=True)
        tail = _normalize(self._buffer)
        self._buffer = ""
        if tail:
            segments.append(tail)
        return segments

    def take(self, limit: int) -> str | None:
        """Force out at most roughly *limit* characters of pending text.

        Used only as a safety valve for streams that never produce punctuation.
        The cut prefers a clause boundary so the fragment is as coherent as
        possible.
        """

        if len(self._buffer) <= limit:
            return None

        window = self._buffer[:limit]
        cuts = [window.rfind(mark) for mark in (", ", "; ", ": ", " ")]
        cut = max(cuts)
        if cut <= 0:
            cut = limit

        piece = _normalize(self._buffer[:cut])
        self._buffer = self._buffer[cut:].lstrip()
        return piece or None

    def _pop(self, *, final: bool) -> list[str]:
        text = self._buffer
        if not text:
            return []

        segments: list[str] = []
        start = 0
        i = 0

        while i < len(text):
            char = text[i]
            if char not in _TERMINATORS:
                i += 1
                continue

            end = self._boundary_end(text, i, final=final)
            if end is not None and self._is_sentence_end(text, i):
                segment = _normalize(text[start:end])
                if segment:
                    segments.append(segment)

                next_start = end
                while next_start < len(text) and text[next_start].isspace():
                    next_start += 1
                start = next_start
                i = next_start
                continue

            # Skip a run such as "?!" or "..." without re-testing every character.
            i += 1
            while i < len(text) and text[i] in _TERMINATORS:
                i += 1

        self._buffer = text[start:]
        return segments

    @staticmethod
    def _boundary_end(text: str, index: int, *, final: bool) -> int | None:
        """Return the exclusive end of a potential sentence, or None."""

        end = index
        while end < len(text) and text[end] in _TERMINATORS:
            end += 1
        while end < len(text) and text[end] in _CLOSERS:
            end += 1

        if end == len(text):
            # Without trailing whitespace we only cut on an explicit flush, unless
            # the caller already signalled that no more text is coming.
            return end if final else None

        if not text[end].isspace():
            return None

        lookahead = end
        while lookahead < len(text) and text[lookahead].isspace():
            lookahead += 1

        if lookahead == len(text):
            # A terminator followed by whitespace is a strong signal that the
            # sentence is done, even if the next sentence has not arrived yet.
            return end

        following = text[lookahead]
        if following.isupper() or following.isdigit() or following in _CLOSERS:
            return end
        if following in _OPENERS:
            return end
        return None

    @staticmethod
    def _is_sentence_end(text: str, index: int) -> bool:
        prefix = text[:index]
        match = _PRECEDING_TOKEN.search(prefix)
        if match is None:
            return True

        token = match.group(1)
        if text[index] == ".":
            lowered = token.lower()
            if lowered in _ABBREVIATIONS:
                return False

            # "1.000" and "14.30" are thousands separators / times, not stops.
            if token.isdigit() and index + 1 < len(text) and text[index + 1].isdigit():
                return False

            # Initials: "J. K. Rowling".
            if len(token) == 1 and token.isalpha() and token.isupper():
                return False

        return True
