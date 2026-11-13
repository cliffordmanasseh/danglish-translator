#!/usr/bin/env python3
"""Batch-translate Danish sermon sources with the local `agy` CLI.

`agy` is an agent CLI, not a plain completion endpoint.  We ask for one
tab-separated line per input and parse the result.  `--dangerously-skip-permissions`
is required for headless mode because the agent otherwise auto-denies its own
tool requests; the prompt explicitly says not to use tools.

Example
-------
    python experiments/sermon/generate_with_agy.py \
        --input /tmp/sermon/data/sermon_test_30.jsonl \
        --output /tmp/sermon/data/agy_test_30.jsonl \
        --model gemini-3.8-flash-high \
        --batch-size 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

INSTRUCTION = (
    "You are a professional Danish-to-English translator specializing in "
    "Christian sermons. Translate each Danish item into English. Preserve "
    "biblical quotations, theological terminology, names, numbers, and "
    "references. Use a natural spoken preaching style. Return exactly one "
    "line per item in the same order, formatted as: id<TAB>English translation. "
    "Do not add any other text. Do not use tools; answer directly."
)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(row["id"]) for row in read_jsonl(path)}


def translate_batch(model: str, rows: list[dict], timeout: float) -> dict[str, str]:
    items = "\n".join(f"{row['id']}\t{row['source']}" for row in rows)
    prompt = f"{INSTRUCTION}\n\nItems:\n{items}"
    proc = subprocess.run(
        [
            "agy",
            "--dangerously-skip-permissions",
            "--model",
            model,
            f"--print={prompt}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"agy exited {proc.returncode}")
    translations: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            translations[key] = value
    return translations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gemini-3.8-flash-high")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    done = done_ids(args.output)
    pending = [row for row in rows if str(row["id"]) not in done]
    print(f"{len(rows)} rows, {len(done)} done, {len(pending)} pending", file=sys.stderr)
    if not pending:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    failed = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.batch_size):
            original_batch = pending[start : start + args.batch_size]
            batch = original_batch
            collected: dict[str, str] = {}
            for attempt in range(1, args.retries + 2):
                try:
                    translations = translate_batch(args.model, batch, args.timeout)
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    print(f"batch {start}: attempt {attempt} failed: {exc}", file=sys.stderr)
                    time.sleep(2)
                    continue
                collected.update(translations)
                missing = [row for row in original_batch if str(row["id"]) not in collected]
                if missing:
                    print(
                        f"batch {start}: missing {len(missing)} ids on attempt {attempt}",
                        file=sys.stderr,
                    )
                    batch = missing
                    continue
                for row in original_batch:
                    handle.write(
                        json.dumps(
                            {
                                **row,
                                "target": collected[str(row["id"])],
                                "teacher": f"agy:{args.model}",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    accepted += 1
                handle.flush()
                print(
                    f"{accepted}/{len(pending)} accepted (batch {start // args.batch_size + 1})",
                    file=sys.stderr,
                )
                break
            else:
                failed += len(original_batch)
                print(f"batch {start}: giving up on {len(original_batch)} rows", file=sys.stderr)

    print(f"wrote {accepted} rows, failed {failed}", file=sys.stderr)


if __name__ == "__main__":
    main()
