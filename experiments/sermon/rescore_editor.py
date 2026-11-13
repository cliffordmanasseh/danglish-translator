#!/usr/bin/env python3
"""Re-score saved sermon predictions against the independent editor references.

Existing result files carry Qwen-generated references.  This swaps in the
editor translations from ``sermon_test_30_editor.jsonl``, matching by source
text, so the same predictions can be compared on a non-circular reference.

Usage:
    python experiments/sermon/rescore_editor.py \
        --results /tmp/sermon/data/agy-sermon-30.json \
        --editor /tmp/sermon/data/sermon_test_30_editor.jsonl \
        --output /tmp/sermon/data/agy-sermon-30-editor.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--editor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    editor = {
        row["source"]: row["reference"]
        for row in map(json.loads, args.editor.read_text(encoding="utf-8").splitlines())
    }
    data = json.loads(args.results.read_text(encoding="utf-8"))
    predictions = data["predictions"]

    missing = [row["source"] for row in predictions if row["source"] not in editor]
    if missing:
        raise SystemExit(f"{len(missing)} predictions have no editor reference, first: {missing[0][:80]}")

    for row in predictions:
        row["reference"] = editor[row["source"]]

    out = {"model": data.get("model", args.results.stem) + "+editor-ref", "predictions": predictions}
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(predictions)} rows to {args.output}")


if __name__ == "__main__":
    main()
