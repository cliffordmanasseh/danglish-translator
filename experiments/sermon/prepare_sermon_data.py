#!/usr/bin/env python3
"""Turn scraped sermons into sentence-level source files.

Entire sermons are held out for the test split, so the test set measures
generalization to unseen preachers/styles rather than memorized sentences.

Example
-------
    python experiments/sermon/prepare_sermon_data.py \
        --input /tmp/sermon/sermons-modern.jsonl \
                /tmp/sermon/sermons-historical.jsonl \
        --output-dir /tmp/sermon/data
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

_SPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÆØÅ])")


def split_sentences(text: str) -> list[str]:
    text = _SPACE_RE.sub(" ", text).strip()
    parts = _SENTENCE_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def load_sermons(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-every", type=int, default=5, help="hold out every Nth sermon")
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=45)
    parser.add_argument("--min-chars", type=int, default=15)
    parser.add_argument("--max-chars", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sermons = load_sermons(args.input)
    sermons.sort(key=lambda row: row["id"])
    rng = random.Random(args.seed)
    rng.shuffle(sermons)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0

    for index, sermon in enumerate(sermons):
        split = "test" if index % args.test_every == 0 else "train"
        sentences = split_sentences(str(sermon["text"]))
        for sentence_index, sentence in enumerate(sentences):
            words = len(sentence.split())
            if not (args.min_words <= words <= args.max_words):
                skipped += 1
                continue
            if not (args.min_chars <= len(sentence) <= args.max_chars):
                skipped += 1
                continue
            if sentence in seen:
                skipped += 1
                continue
            seen.add(sentence)
            row = {
                "id": f"{sermon['id']}-{sentence_index:04d}",
                "source": sentence,
                "sermon_id": sermon["id"],
                "sermon_title": sermon.get("title", ""),
                "speaker": sermon.get("speaker", ""),
                "year": sermon.get("year", 0),
                "url": sermon.get("url", ""),
                "domain": "sermon",
                "split": split,
            }
            (test_rows if split == "test" else train_rows).append(row)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("sermon_train_sources.jsonl", train_rows),
        ("sermon_test_sources.jsonl", test_rows),
    ):
        path = output_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} rows", file=sys.stderr)

    manifest = {
        "sermons": len(sermons),
        "train_sentences": len(train_rows),
        "test_sentences": len(test_rows),
        "skipped": skipped,
        "test_every": args.test_every,
        "seed": args.seed,
        "train_word_median": int(statistics.median(len(r["source"].split()) for r in train_rows))
        if train_rows
        else 0,
        "test_word_median": int(statistics.median(len(r["source"].split()) for r in test_rows))
        if test_rows
        else 0,
    }
    (output_dir / "sermon_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
