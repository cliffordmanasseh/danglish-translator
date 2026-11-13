#!/usr/bin/env python3
"""Build the pilot source list and the final synthetic + real training file.

The benchmark uses 250 Tatoeba and 250 NTREX pairs sampled with seed 13.  Those
sources are excluded here so the pilot cannot train on the evaluation set.

Typical flow
------------
    # 1. Make disjoint source pools: one for synthetic generation, one for real
    python experiments/prepare_pilot_data.py sources \
        --output-dir /tmp/pilot

    # 2. Generate /tmp/pilot/synthetic.jsonl with generate_synthetic.py

    # 3. Mix synthetic and real pairs
    python experiments/prepare_pilot_data.py combine \
        --output-dir /tmp/pilot
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mt_benchmark import load_ntrex, load_tatoeba  # noqa: E402

DEFAULT_TATOEBA = Path("/tmp/opencode/mt-bench/data/tatoeba")
DEFAULT_NTREX = Path("/tmp/opencode/mt-bench/data")
BENCHMARK_SEED = 13
BENCHMARK_LIMIT = 250


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def benchmark_sources(tatoeba: Path, ntrex: Path) -> set[str]:
    rows = load_tatoeba(tatoeba, BENCHMARK_LIMIT, BENCHMARK_SEED)
    rows += load_ntrex(ntrex, BENCHMARK_LIMIT, BENCHMARK_SEED)
    return {row["source"] for row in rows}


def command_sources(args: argparse.Namespace) -> None:
    tatoeba = Path(args.tatoeba)
    ntrex = Path(args.ntrex)
    held_out = benchmark_sources(tatoeba, ntrex)

    all_tatoeba = [
        row for row in load_tatoeba(tatoeba, 10**9, args.seed) if row["source"] not in held_out
    ]
    rng = random.Random(args.seed)
    rng.shuffle(all_tatoeba)

    synthetic = all_tatoeba[: args.synthetic_limit]
    real_tatoeba = all_tatoeba[args.synthetic_limit : args.synthetic_limit + args.real_limit]
    real_ntrex = [
        row for row in load_ntrex(ntrex, 10**9, args.seed) if row["source"] not in held_out
    ]

    source_rows = [
        {
            "id": f"tatoeba-{index:06d}",
            "source": row["source"],
            "domain": "tatoeba",
        }
        for index, row in enumerate(synthetic)
    ]
    real_rows = [
        {
            "id": f"tatoeba-real-{index:06d}",
            "source": row["source"],
            "target": row["reference"],
            "domain": "tatoeba",
            "kind": "real",
        }
        for index, row in enumerate(real_tatoeba)
    ]
    real_rows += [
        {
            "id": f"ntrex-real-{index:06d}",
            "source": row["source"],
            "target": row["reference"],
            "domain": "ntrex",
            "kind": "real",
        }
        for index, row in enumerate(real_ntrex)
    ]

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "pilot_sources.jsonl", source_rows)
    write_jsonl(output_dir / "pilot_real.jsonl", real_rows)
    manifest = {
        "seed": args.seed,
        "held_out_benchmark": len(held_out),
        "tatoeba_eligible": len(all_tatoeba),
        "synthetic_sources": len(source_rows),
        "real_tatoeba": len(real_tatoeba),
        "real_ntrex": len(real_ntrex),
        "total_real": len(real_rows),
    }
    (output_dir / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def command_combine(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    synthetic_path = Path(args.synthetic) if args.synthetic else output_dir / "synthetic.jsonl"
    real_path = Path(args.real) if args.real else output_dir / "pilot_real.jsonl"

    synthetic = [
        {
            "id": row.get("id", f"syn-{index:06d}"),
            "source": row["source"],
            "target": row["target"],
            "domain": row.get("domain", "unknown"),
            "kind": "synthetic",
        }
        for index, row in enumerate(read_jsonl(synthetic_path))
        if row.get("source") and row.get("target")
    ]
    if args.max_synthetic:
        synthetic = synthetic[: args.max_synthetic]
    real = [
        {
            "id": row.get("id", f"real-{index:06d}"),
            "source": row["source"],
            "target": row["target"],
            "domain": row.get("domain", "unknown"),
            "kind": "real",
        }
        for index, row in enumerate(read_jsonl(real_path))
        if row.get("source") and row.get("target")
    ]
    if args.max_real:
        random.Random(args.seed).shuffle(real)
        real = real[: args.max_real]

    # Keep one row per source.  Synthetic wins over real when both exist.
    seen: set[str] = set()
    combined: list[dict[str, Any]] = []
    for row in synthetic + real:
        if row["source"] in seen:
            continue
        seen.add(row["source"])
        combined.append(row)
    random.Random(args.seed).shuffle(combined)
    if args.max_total:
        combined = combined[: args.max_total]

    output = Path(args.output) if args.output else output_dir / "pilot_train.jsonl"
    write_jsonl(output, combined)
    manifest = {
        "synthetic": len(synthetic),
        "real": len(real),
        "combined": len(combined),
        "max_synthetic": args.max_synthetic,
        "max_real": args.max_real,
        "output": str(output),
        "seed": args.seed,
    }
    (output_dir / "pilot_train_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sources")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tatoeba", default=str(DEFAULT_TATOEBA))
    p.add_argument("--ntrex", default=str(DEFAULT_NTREX))
    p.add_argument("--synthetic-limit", type=int, default=12000)
    p.add_argument("--real-limit", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=command_sources)

    p = sub.add_parser("combine")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--synthetic")
    p.add_argument("--real")
    p.add_argument("--output")
    p.add_argument("--max-synthetic", type=int, default=0)
    p.add_argument("--max-real", type=int, default=0)
    p.add_argument("--max-total", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=command_combine)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
