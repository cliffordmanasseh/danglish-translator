"""CTranslate2 backend for Marian / OPUS-MT models."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from ..errors import BackendError, DependencyMissingError, ModelNotFoundError

_REQUIRED_FILES = ("model.bin", "source.spm", "target.spm")
_MAX_INPUT_TOKENS = 500
_BATCH_SIZE = 16
# Marian's Hugging Face tokenizer appends this end-of-sentence token to the source.
# CTranslate2 does not do it automatically, and without it these models produce
# severe repetition loops. This single token is the difference between usable and
# unusable output.
_SOURCE_EOS = "</s>"


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


class CT2Backend:
    """Quantized CTranslate2 translation backend.

    The model directory is the output of ``danglish fetch`` (or any compatible
    CTranslate2 Marian conversion) and must contain ``model.bin``, ``source.spm``
    and ``target.spm``.
    """

    name = "ct2"

    def __init__(
        self,
        model_dir: Path | str,
        *,
        compute_type: str = "int8",
        device: str = "cpu",
        intra_threads: int = 0,
        inter_threads: int = 1,
        beam_size: int = 2,
        max_decoding_length: int = 256,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self.max_decoding_length = max_decoding_length
        self.intra_threads = intra_threads or min(4, os.cpu_count() or 4)

        missing = [name for name in _REQUIRED_FILES if not (self.model_dir / name).is_file()]
        if missing:
            listing = ", ".join(missing)
            raise ModelNotFoundError(
                f"{self.model_dir} is not a usable CTranslate2 model (missing: {listing}).\n"
                "Run `danglish fetch` to download and convert the default model, or pass "
                "`--model-path` to an existing conversion."
            )

        try:
            import ctranslate2
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover - core dependency is normally present
            raise DependencyMissingError(
                "ctranslate2 and sentencepiece are required for translation. "
                'Install them with: pip install "danglish"'
            ) from exc

        try:
            self._translator = ctranslate2.Translator(
                str(self.model_dir),
                device=device,
                compute_type=compute_type,
                intra_threads=self.intra_threads,
                inter_threads=inter_threads,
            )
        except (ValueError, RuntimeError) as exc:
            raise BackendError(
                f"could not load CTranslate2 model from {self.model_dir}: {exc}"
            ) from exc

        try:
            self._source_sp = spm.SentencePieceProcessor(
                model_file=str(self.model_dir / "source.spm")
            )
            self._target_sp = spm.SentencePieceProcessor(
                model_file=str(self.model_dir / "target.spm")
            )
        except (OSError, RuntimeError) as exc:
            raise BackendError(f"could not load SentencePiece models: {exc}") from exc

    def __repr__(self) -> str:
        return (
            f"CT2Backend(model_dir={str(self.model_dir)!r}, compute_type={self.compute_type!r}, "
            f"threads={self.intra_threads})"
        )

    def translate(self, texts: Sequence[str], *, beam_size: int | None = None) -> list[str]:
        if isinstance(texts, str):  # defensive: the protocol is sequence-based
            texts = [texts]
        cleaned = [_normalize_whitespace(text) for text in texts]
        if not cleaned:
            return []

        expanded: list[str] = []
        owners: list[int] = []
        for index, text in enumerate(cleaned):
            if not text:
                continue
            for chunk in self._split_long_text(text):
                expanded.append(chunk)
                owners.append(index)

        if not expanded:
            return ["" for _ in cleaned]

        token_batches = [
            [*self._source_sp.encode(text, out_type=str), _SOURCE_EOS] for text in expanded
        ]
        try:
            results = self._translator.translate_batch(
                token_batches,
                beam_size=beam_size or self.beam_size,
                max_batch_size=min(_BATCH_SIZE, len(token_batches)),
                max_input_length=_MAX_INPUT_TOKENS,
                max_decoding_length=self.max_decoding_length,
                return_scores=False,
            )
        except (ValueError, RuntimeError) as exc:
            raise BackendError(f"translation failed: {exc}") from exc

        parts: list[list[str]] = [[] for _ in cleaned]
        for owner, result in zip(owners, results, strict=True):
            text = self._decode(result.hypotheses[0] if result.hypotheses else [])
            if text:
                parts[owner].append(text)

        return [" ".join(chunks).strip() for chunks in parts]

    def _decode(self, tokens: list[str]) -> str:
        # SentencePiece marks end-of-sentence and padding as control tokens. They
        # should not be rendered into the translated text.
        cleaned = [token for token in tokens if token not in {"</s>", "<pad>", "<unk>"}]
        if not cleaned:
            return ""
        return self._target_sp.decode(cleaned).strip()

    def _split_long_text(self, text: str) -> list[str]:
        """Split unusually long input so the model never sees more than 500 tokens.

        Normal streaming input is already sentence-sized. This is a safety net for
        one-shot translations of long paragraphs.
        """

        try:
            token_count = len(self._source_sp.encode(text, out_type=str))
        except (RuntimeError, TypeError):
            token_count = 0
        if token_count <= _MAX_INPUT_TOKENS or len(text.split()) < 2:
            return [text]

        words = text.split()
        midpoint = len(words) // 2
        left = " ".join(words[:midpoint])
        right = " ".join(words[midpoint:])
        return [*self._split_long_text(left), *self._split_long_text(right)]

    def warmup(self) -> None:
        self.translate(["Hej"], beam_size=1)

    def close(self) -> None:
        # Dropping the reference is enough; CTranslate2 does not expose a portable
        # explicit unload across all supported versions.
        self._translator = None  # type: ignore[assignment]
        self._source_sp = None  # type: ignore[assignment]
        self._target_sp = None  # type: ignore[assignment]
