#!/usr/bin/env python3
"""Combine sermon teacher output with general replay data for LoRA training.

Priority for duplicate sources is ``agy`` > ``qwen`` > ``general``.  The 200
``agy`` pairs cover the first 200 sermon sources, so their Qwen duplicates are
dropped.  General pairs are kept as replay to limit catastrophic forgetting.

Usage:
    python experiments/sermon/combine_train_data.py \
        --qwen /tmp/sermon/data/qwen_train.jsonl \
        --agy /tmp/sermon/data/agy_train.jsonl \
        --general /tmp/sermon/data/sermon_agy_mix.jsonl \
        --output /tmp/sermon/data/sermon_qwen_agy_mix.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_source(source: str) -> str:
    return " ".join(source.split())


def prepare(rows: list[dict], kind: str, domain: str | None = None) -> list[dict]:
    out: list[dict] = []
    for index, row in enumerate(rows):
        source = str(row.get("source", "")).strip()
        target = str(row.get("target", "")).strip()
        if not source or not target:
            continue
        if domain is not None and row.get("domain") != domain:
            continue
        out.append(
            {
                "id": row.get("id", f"{kind}-{index:06d}"),
                "source": source,
                "target": target,
                "domain": row.get("domain", kind),
                "kind": kind,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--agy", type=Path, required=True)
    parser.add_argument("--general", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-qwen", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    qwen = prepare(read_jsonl(args.qwen), "qwen", domain="sermon")
    agy = prepare(read_jsonl(args.agy), "agy", domain="sermon")
    general = prepare(read_jsonl(args.general), "general")
    general = [row for row in general if row["domain"] != "sermon"]
    if args.max_qwen:
        qwen = qwen[: args.max_qwen]

    seen: set[str] = set()
    combined: list[dict] = []
    for row in agy + qwen + general:
        key = normalize_source(row["source"])
        if key in seen:
            continue
        seen.add(key)
        combined.append(row)

    random.Random(args.seed).shuffle(combined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(row["kind"] for row in combined)
    manifest = {
        "output": str(args.output),
        "total": len(combined),
        "by_kind": dict(counts),
        "by_domain": dict(Counter(row["domain"] for row in combined)),
        "seed": args.seed,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
