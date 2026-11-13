#!/usr/bin/env python3
"""Filter teacher-generated pairs before training.

This is stricter than the generator's rejection pass. It removes structural
failures and fragments that are not useful translation examples, while keeping
the original rows intact for inspection.

Usage:
    python experiments/sermon/filter_generated_pairs.py \
        --input data/sermons/qwen_train_full.jsonl \
        --output data/sermons/qwen_train_clean.jsonl \
        --rejected data/sermons/qwen_train_rejected.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

DANISH_WORDS = {
    "og", "ikke", "der", "som", "jeg", "vi", "til", "på", "er", "med",
    "af", "den", "det", "en", "et", "har", "kan", "skal", "være", "blev",
    "bliver", "også", "men", "så", "hvad", "hvor", "når", "vores", "deres",
}
FRAGMENT_END_RE = re.compile(r"[,;:–—-]\s*$")
REFERENCE_RE = re.compile(
    r"(?:\bchr\.|\bcap\.|\bvers\b|\bv\.\s*\d|\b\d+[.,:]\s*\d+|/\s*(?:oc|og)\s+\d+)",
    re.IGNORECASE,
)


def repeated_ngram(text: str, n: int = 4) -> int:
    words = text.lower().split()
    if len(words) < n:
        return 0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return Counter(grams).most_common(1)[0][1] if grams else 0


def rejection_reason(row: dict, args: argparse.Namespace) -> str | None:
    source = str(row.get("source", "")).strip()
    target = str(row.get("target", "")).strip()
    source_words = len(source.split())
    target_words = len(target.split())
    ratio = target_words / source_words if source_words else 0.0

    if not target:
        return "empty"
    if target == source:
        return "copy"
    if "<|im_start|>" in target or "<|im_end|>" in target:
        return "special_token"
    if row.get("stopped") is False:
        return "truncated"
    if source_words < args.min_source_words:
        return "source_too_short"
    if target_words < args.min_target_words:
        return "target_too_short"
    if ratio < args.min_ratio or ratio > args.max_ratio:
        return f"ratio:{ratio:.2f}"
    if FRAGMENT_END_RE.search(source):
        return "source_fragment"
    if FRAGMENT_END_RE.search(target):
        return "target_fragment"
    if source_words < 10 and REFERENCE_RE.search(source):
        return "source_reference"
    if repeated_ngram(target) > args.max_repeat:
        return "repetition"

    words = target.lower().split()
    if words:
        danish_rate = sum(
            1 for word in words if word.strip(".,!?;:»«\"'") in DANISH_WORDS
        ) / len(words)
        if danish_rate > args.danish_threshold:
            return "danish_residue"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--min-ratio", type=float, default=0.55)
    parser.add_argument("--max-ratio", type=float, default=2.0)
    parser.add_argument("--min-source-words", type=int, default=4)
    parser.add_argument("--min-target-words", type=int, default=3)
    parser.add_argument("--max-repeat", type=int, default=4)
    parser.add_argument("--danish-threshold", type=float, default=0.20)
    args = parser.parse_args()

    reasons: Counter[str] = Counter()
    kept = 0
    rejected = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out, args.rejected.open(
        "w", encoding="utf-8"
    ) as bad:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            reason = rejection_reason(row, args)
            if reason is None:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            else:
                reasons[reason.split(":")[0]] += 1
                bad.write(json.dumps({**row, "filter_reason": reason}, ensure_ascii=False) + "\n")
                rejected += 1

    print(
        json.dumps(
            {
                "input": str(args.input),
                "kept": kept,
                "rejected": rejected,
                "rejected_rate": round(rejected / max(1, kept + rejected), 4),
                "reasons": dict(reasons),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
