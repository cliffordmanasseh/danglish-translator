#!/usr/bin/env python3
"""Attach independent editor translations to the 30-pair sermon test set.

The machine references in ``sermon_test_30.jsonl`` were produced by Qwen3.8
Flash.  Evaluating a Qwen-trained student against them is circular, so these
30 sentences were translated directly from the Danish without consulting the
machine references.  This script joins the editor translations by row index.

Usage:
    python experiments/sermon/build_editor_references.py \
        --test /tmp/sermon/data/sermon_test_30.jsonl \
        --translations /tmp/sermon/data/editor_translations_30.tsv \
        --output /tmp/sermon/data/sermon_test_30_editor.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def read_translations(path: Path) -> dict[int, str]:
    translations: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        index, text = line.split("\t", 1)
        translations[int(index)] = text.strip()
    return translations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.test)
    translations = read_translations(args.translations)
    if len(rows) != len(translations):
        raise SystemExit(f"row mismatch: {len(rows)} test rows, {len(translations)} translations")

    out = []
    for index, row in enumerate(rows):
        if index not in translations:
            raise SystemExit(f"missing translation for row {index}")
        out.append(
            {
                "id": row.get("id", f"sermon-{index:04d}"),
                "source": row["source"],
                "reference": translations[index],
                "reference_kind": "independent_editor",
                "domain": "sermon",
                "sermon_id": row.get("sermon_id"),
                "speaker": row.get("speaker"),
                "year": row.get("year"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out),
        encoding="utf-8",
    )
    print(f"wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
