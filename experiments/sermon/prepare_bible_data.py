#!/usr/bin/env python3
"""Build a CC0 Bible verse-pair training file from OPUS bible-uedin.

Download:
    curl -L -o /tmp/sermon/bible/da-en.zip \
      https://object.pouta.csc.fi/OPUS-bible-uedin/v1/moses/da-en.txt.zip
    unzip -o /tmp/sermon/bible/da-en.zip -d /tmp/sermon/bible

The corpus is CC0 1.0.  The Danish/English translations are older than modern
sermon language, so mix this with modern general data rather than training on
Bible verses alone.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da", required=True, type=Path)
    parser.add_argument("--en", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=50)
    args = parser.parse_args()

    da = [line.strip() for line in args.da.read_text(encoding="utf-8").splitlines()]
    en = [line.strip() for line in args.en.read_text(encoding="utf-8").splitlines()]
    if len(da) != len(en):
        raise SystemExit(f"unaligned files: {len(da)} vs {len(en)}")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, target in zip(da, en, strict=True):
        if not source or not target or source in seen:
            continue
        source_words = len(source.split())
        target_words = len(target.split())
        if not (args.min_words <= source_words <= args.max_words):
            continue
        if not (args.min_words <= target_words <= args.max_words):
            continue
        seen.add(source)
        rows.append(
            {
                "id": f"bible-{len(rows):06d}",
                "source": source,
                "target": target,
                "domain": "bible",
                "kind": "real",
                "license": "CC0-1.0",
            }
        )

    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} verse pairs to {args.output}")


if __name__ == "__main__":
    main()
