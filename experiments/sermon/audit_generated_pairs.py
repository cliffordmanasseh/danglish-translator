#!/usr/bin/env python3
"""Audit teacher-generated sermon pairs for structural and terminology issues.

This is a tail-risk filter, not a semantic proof. It flags rows for review
rather than silently deleting them, and reports corpus-level rates so the
teacher output can be compared across runs.

Checks:
- empty / copy / special tokens
- target/source word ratio
- truncation (`stopped` false)
- Danish residue (function-word rate in the English target)
- repeated 4-grams
- missing numbers from the source
- glossary opportunities and hits

Usage:
    python experiments/sermon/audit_generated_pairs.py \
        --input data/sermons/qwen_train_full.jsonl \
        --output data/sermons/qwen_train_full.audit.json \
        --flagged data/sermons/qwen_train_full.flagged.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

DANISH_WORDS = {
    "og", "ikke", "der", "som", "jeg", "vi", "til", "på", "er", "med",
    "af", "den", "det", "en", "et", "har", "kan", "skal", "være", "blev",
    "bliver", "også", "men", "så", "hvad", "hvor", "når", "vores", "deres",
}
NUMBER_RE = re.compile(r"\d+")


def load_glossary(path: Path) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        danish = parts[0].strip()
        english = [part.strip().lower() for part in parts[1].split("|") if part.strip()]
        if danish and english:
            rows.append((danish, english))
    return sorted(rows, key=lambda item: len(item[0]), reverse=True)


def term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def repeated_ngram(text: str, n: int = 4) -> int:
    words = text.lower().split()
    if len(words) < n:
        return 0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return Counter(grams).most_common(1)[0][1] if grams else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--glossary", type=Path, default=Path("experiments/sermon/glossary.tsv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flagged", type=Path, required=True)
    parser.add_argument("--min-ratio", type=float, default=0.55)
    parser.add_argument("--max-ratio", type=float, default=2.0)
    parser.add_argument("--danish-threshold", type=float, default=0.20)
    args = parser.parse_args()

    glossary = load_glossary(args.glossary)
    reasons: Counter[str] = Counter()
    ratios: list[float] = []
    opportunities = 0
    glossary_hits = 0
    glossary_misses: list[dict[str, str]] = []
    flagged_rows: list[dict] = []
    rows = 0

    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        row_reasons: list[str] = []

        source_words = len(source.split())
        target_words = len(target.split())
        ratio = target_words / source_words if source_words else 0.0
        ratios.append(ratio)

        if not target.strip():
            row_reasons.append("empty")
        if target.strip() == source.strip():
            row_reasons.append("copy")
        if "<|im_start|>" in target or "<|im_end|>" in target:
            row_reasons.append("special_token")
        if row.get("stopped") is False:
            row_reasons.append("truncated")
        if ratio < args.min_ratio or ratio > args.max_ratio:
            row_reasons.append("ratio")

        words = target.lower().split()
        if words:
            danish_rate = sum(1 for word in words if word.strip(".,!?;:»«\"'") in DANISH_WORDS) / len(words)
            if danish_rate > args.danish_threshold:
                row_reasons.append("danish_residue")

        if repeated_ngram(target) > 2:
            row_reasons.append("repetition")

        missing_numbers = [
            number for number in set(NUMBER_RE.findall(source)) if number not in target
        ]
        if missing_numbers:
            row_reasons.append("number")

        lowered = target.lower()
        for danish, english in glossary:
            if not term_pattern(danish).search(source):
                continue
            opportunities += 1
            if any(term in lowered for term in english):
                glossary_hits += 1
            else:
                if len(glossary_misses) < 50:
                    glossary_misses.append(
                        {
                            "source": source[:200],
                            "target": target[:200],
                            "danish": danish,
                            "expected": "|".join(english),
                        }
                    )

        for reason in row_reasons:
            reasons[reason] += 1
        if row_reasons:
            flagged_rows.append(
                {
                    "id": row.get("id"),
                    "reasons": row_reasons,
                    "ratio": round(ratio, 3),
                    "source": source,
                    "target": target,
                }
            )

    ratios.sort()
    summary = {
        "rows": rows,
        "flagged": len(flagged_rows),
        "flagged_rate": round(len(flagged_rows) / rows, 4) if rows else 0.0,
        "reasons": dict(reasons),
        "ratio_p10": round(ratios[len(ratios) // 10], 3) if ratios else 0.0,
        "ratio_p50": round(statistics.median(ratios), 3) if ratios else 0.0,
        "ratio_p90": round(ratios[9 * len(ratios) // 10], 3) if ratios else 0.0,
        "glossary_opportunities": opportunities,
        "glossary_hits": glossary_hits,
        "glossary_accuracy": round(glossary_hits / opportunities, 4) if opportunities else 0.0,
        "glossary_misses": glossary_misses,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.flagged.parent.mkdir(parents=True, exist_ok=True)
    with args.flagged.open("w", encoding="utf-8") as handle:
        for row in flagged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "glossary_misses"}, indent=2))


if __name__ == "__main__":
    main()
