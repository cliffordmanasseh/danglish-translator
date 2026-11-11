"""Exceptions raised by Danglish."""

from __future__ import annotations


class DanglishError(Exception):
    """Base class for all user-facing Danglish errors."""


class ConfigError(DanglishError):
    """The configuration file is missing or invalid."""


class ModelNotFoundError(DanglishError):
    """The converted CTranslate2 model could not be found."""


class DependencyMissingError(DanglishError):
    """An optional dependency is required for the requested feature."""


class BackendError(DanglishError):
    """The translation backend failed to load or run."""
