#!/usr/bin/env python3
"""Join sermon test sources with teacher-generated references.

The reference is a machine translation, not a human reference.  Use it for a
first signal and commission human references before a release.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sources = read_jsonl(args.sources)
    targets = read_jsonl(args.targets)
    pairs = [
        {
            "source": row["source"],
            "reference": targets[key]["target"],
            "domain": "sermon",
            "id": key,
            "sermon_id": row.get("sermon_id", ""),
            "speaker": row.get("speaker", ""),
            "year": row.get("year", 0),
        }
        for key, row in sources.items()
        if key in targets
    ]
    random.Random(args.seed).shuffle(pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} test pairs to {args.output}")


if __name__ == "__main__":
    main()
