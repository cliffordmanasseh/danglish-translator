#!/usr/bin/env python3
"""Filter generated pairs by their round-trip chrF++ score.

Use this after `roundtrip_audit.py --scores ...` has scored the corpus (or a
subset). Rows below the threshold are written to a rejected file with the score
and source/target for review. Rows without a score are kept by default so a
partial scoring pass cannot silently delete data.

Usage:
    python experiments/sermon/filter_by_roundtrip.py \
        --input data/sermons/qwen_train_full.jsonl \
        --scores data/sermons/roundtrip_full_scores.jsonl \
        --output data/sermons/qwen_train_roundtrip_clean.jsonl \
        --rejected data/sermons/qwen_train_roundtrip_rejected.jsonl \
        --min-score 25
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--min-score", type=float, default=25.0)
    parser.add_argument("--drop-missing", action="store_true")
    args = parser.parse_args()

    scores = {}
    for line in args.scores.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scores[row["id"]] = row

    meta = {}
    if args.metadata:
        meta = {
            row["id"]: row
            for row in map(json.loads, args.metadata.read_text(encoding="utf-8").splitlines())
        }

    kept = 0
    rejected = 0
    missing = 0
    by_decade_kept: dict[str, list[float]] = defaultdict(list)
    by_decade_rejected: dict[str, list[float]] = defaultdict(list)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out, args.rejected.open(
        "w", encoding="utf-8"
    ) as bad:
        for line in args.input.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            scored = scores.get(row.get("id", ""))
            year = int(meta.get(row.get("id", ""), {}).get("year", 0))
            decade = f"{(year // 10) * 10}s" if year else "unknown"

            if scored is None:
                missing += 1
                if args.drop_missing:
                    bad.write(json.dumps({**row, "filter_reason": "missing_score"}, ensure_ascii=False) + "\n")
                    rejected += 1
                    continue
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                continue

            score = float(scored["score"])
            if score < args.min_score:
                by_decade_rejected[decade].append(score)
                bad.write(
                    json.dumps(
                        {**row, "roundtrip_chrf++": score, "filter_reason": "low_roundtrip"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rejected += 1
            else:
                by_decade_kept[decade].append(score)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1

    summary = {
        "kept": kept,
        "rejected": rejected,
        "missing_scores": missing,
        "min_score": args.min_score,
        "kept_mean_score": round(statistics.mean([s for v in by_decade_kept.values() for s in v]), 2)
        if by_decade_kept
        else 0.0,
        "by_decade_kept": {k: len(v) for k, v in sorted(by_decade_kept.items())},
        "by_decade_rejected": {k: len(v) for k, v in sorted(by_decade_rejected.items())},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
