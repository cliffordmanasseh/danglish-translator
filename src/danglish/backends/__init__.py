"""Backend factory."""

from __future__ import annotations

from ..config import AppConfig
from .base import TranslatorBackend
from .mock import MockBackend


def create_backend(config: AppConfig, *, mock: bool = False) -> TranslatorBackend:
    if mock:
        return MockBackend()

    # Imported lazily so `danglish doctor` and the mock demo stay fast.
    from .ct2 import CT2Backend

    return CT2Backend(
        config.model.path,
        compute_type=config.model.compute_type,
        device=config.model.device,
        intra_threads=config.model.intra_threads,
        inter_threads=config.model.inter_threads,
        beam_size=config.model.beam_size,
        max_decoding_length=config.model.max_decoding_length,
    )


__all__ = ["MockBackend", "TranslatorBackend", "create_backend"]
