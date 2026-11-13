#!/usr/bin/env python3
"""Build a clean, deduplicated sermon corpus from scraped JSONL files.

The scraper keeps one row per sermon with the raw body.  This step normalizes
whitespace, drops very short or duplicate texts, adds a content hash and word
count, and writes a manifest with counts by license and decade.

Usage:
    python experiments/sermon/build_sermon_corpus.py \
        data/sermons/dansketaler_sermons_modern.jsonl \
        data/sermons/dansketaler_sermons_historical.jsonl \
        --output data/sermons/corpus.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def decade(year: int) -> str:
    if not year:
        return "unknown"
    return f"{(year // 10) * 10}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-chars", type=int, default=500)
    args = parser.parse_args()

    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()
    rows: list[dict] = []
    skipped_short = 0
    skipped_duplicate = 0

    for path in args.inputs:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = normalize_text(str(row.get("text", "")))
            if len(text) < args.min_chars:
                skipped_short += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            url = str(row.get("url", ""))
            if url in seen_urls or digest in seen_hashes:
                skipped_duplicate += 1
                continue
            seen_urls.add(url)
            seen_hashes.add(digest)

            year = int(row.get("year") or 0)
            words = len(text.split())
            rows.append(
                {
                    "id": row.get("id") or digest[:16],
                    "url": url,
                    "title": row.get("title", ""),
                    "year": year,
                    "decade": decade(year),
                    "speaker": row.get("speaker", ""),
                    "location": row.get("location", ""),
                    "source_note": row.get("source_note", ""),
                    "license": row.get("license", "copyright_unknown"),
                    "text_sha256": digest,
                    "chars": len(text),
                    "words": words,
                    "text": text,
                }
            )

    rows.sort(key=lambda row: (row["year"], row["title"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_license = Counter(row["license"] for row in rows)
    by_decade = Counter(row["decade"] for row in rows)
    manifest = {
        "output": str(args.output),
        "sermons": len(rows),
        "words": sum(row["words"] for row in rows),
        "chars": sum(row["chars"] for row in rows),
        "skipped_short": skipped_short,
        "skipped_duplicate": skipped_duplicate,
        "by_license": dict(by_license),
        "by_decade": dict(sorted(by_decade.items())),
        "earliest_year": min((row["year"] for row in rows if row["year"]), default=0),
        "latest_year": max((row["year"] for row in rows), default=0),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
