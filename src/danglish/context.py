"""Translation context profiles.

The compact OPUS-MT model cannot accept free-form instructions like “this is a
sermon”. A profile therefore does two things it *can* do:

* apply terminology preferences to the English output, but only when the Danish
  source actually contains the corresponding term;
* reserve a place for a future domain-specific model path.

This is post-processing, not model conditioning. It improves consistency for known
terms; it does not give the model broader discourse context. See docs/context.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .errors import DanglishError


@dataclass(frozen=True, slots=True)
class GlossaryRule:
    """Prefer one English term when the Danish source contains another term."""

    source_terms: tuple[str, ...]
    target_patterns: tuple[str, ...]
    replacement: str
    preserve_case: bool = True
    only_if_unique: bool = True
    blocked_source_terms: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True, slots=True)
class ContextProfile:
    name: str
    label: str
    description: str
    glossary: tuple[GlossaryRule, ...] = ()
    model_path: Path | None = None

    def apply(self, source: str, target: str) -> str:
        """Apply this profile's terminology preferences to one translation."""

        if not self.glossary or not target:
            return target

        source_lower = source.casefold()
        updated = target
        for rule in self.glossary:
            if not any(term.casefold() in source_lower for term in rule.source_terms):
                continue
            if any(term.casefold() in source_lower for term in rule.blocked_source_terms):
                continue
            for pattern in rule.target_patterns:
                regex = re.compile(
                    rf"(?<!\w){re.escape(pattern)}(?!\w)",
                    re.IGNORECASE,
                )
                matches = list(regex.finditer(updated))
                if not matches:
                    continue
                # A blanket replacement is dangerous when the same English word
                # appears twice and only one instance refers to the source term.
                if rule.only_if_unique and len(matches) > 1:
                    continue
                updated = regex.sub(
                    lambda match, current=rule: _replace(match.group(0), current),
                    updated,
                )
                # Patterns on a rule are alternatives. Stop after the first
                # successful replacement so a shorter pattern cannot rewrite the
                # replacement itself (e.g. "the Holy Spirit" -> "the the ...").
                break
        return updated


def _replace(original: str, rule: GlossaryRule) -> str:
    if not rule.preserve_case:
        return rule.replacement if not original.isupper() else rule.replacement.upper()
    if original.isupper():
        return rule.replacement.upper()
    if original[:1].isupper():
        return rule.replacement[:1].upper() + rule.replacement[1:]
    return rule.replacement


_GENERAL = ContextProfile(
    name="general",
    label="General",
    description="Neutral everyday translation with no terminology overrides.",
)

_SERMON = ContextProfile(
    name="sermon",
    label="Church / sermon",
    description=(
        "Prefers congregational terminology (congregation, sermon, pastor, worship "
        "service, Holy Communion). Apply it when the source is a sermon or church text."
    ),
    glossary=(
        GlossaryRule(
            source_terms=("menighed",),
            target_patterns=("church", "parish"),
            replacement="congregation",
            blocked_source_terms=("kirke",),
            note="menighed → congregation, not the building",
        ),
        GlossaryRule(
            source_terms=("prædiken",),
            target_patterns=("sermon", "homily", "preaching"),
            replacement="sermon",
        ),
        GlossaryRule(
            source_terms=("præst",),
            target_patterns=("priest", "minister", "clergyman", "vicar"),
            replacement="pastor",
        ),
        GlossaryRule(
            source_terms=("gudstjeneste",),
            target_patterns=("service", "worship service", "church service"),
            replacement="worship service",
        ),
        GlossaryRule(
            source_terms=("nadver",),
            target_patterns=("the lord's supper", "lord's supper", "communion", "eucharist"),
            replacement="Holy Communion",
            preserve_case=False,
        ),
        GlossaryRule(
            source_terms=("helligånd",),
            target_patterns=("the holy spirit", "the holy ghost", "holy spirit", "holy ghost"),
            replacement="the Holy Spirit",
            preserve_case=False,
        ),
        GlossaryRule(
            source_terms=("frelser",),
            target_patterns=("savior", "saviour"),
            replacement="Savior",
            preserve_case=False,
        ),
    ),
)

_PROFILES: dict[str, ContextProfile] = {
    _GENERAL.name: _GENERAL,
    _SERMON.name: _SERMON,
}


def profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)


def list_profiles() -> tuple[ContextProfile, ...]:
    return tuple(_PROFILES.values())


def get_profile(name: str | None) -> ContextProfile:
    if not name:
        return _GENERAL
    try:
        return _PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(_PROFILES)
        raise DanglishError(f"unknown context profile {name!r}; choose one of: {choices}") from exc
