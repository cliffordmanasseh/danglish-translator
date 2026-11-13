#!/usr/bin/env python3
"""Add COMET scores to one or more mt_benchmark result JSONs.

COMET is intentionally separate from ``mt_benchmark.py``: it pulls a large
checkpoint and a Lightning stack, and most quick checks only need chrF++/BLEU.

Example
-------
    python experiments/score_comet.py \
        experiments/results/opus-big.json \
        experiments/results/eurollm-ft.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_comet(model_name: str):
    from comet import download_model, load_from_checkpoint

    path = download_model(model_name)
    return load_from_checkpoint(path)


def score_file(model: Any, path: Path, batch_size: int, gpus: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["predictions"]
    payload = [
        {"src": row["source"], "mt": row["prediction"], "ref": row["reference"]} for row in rows
    ]
    output = model.predict(payload, batch_size=batch_size, gpus=gpus, progress_bar=False)
    for row, score in zip(rows, output.scores, strict=True):
        row["comet"] = round(float(score), 4)

    metrics = data.setdefault("metrics", {})
    metrics["comet"] = round(float(output.system_score), 4)
    for domain in sorted({row["domain"] for row in rows}):
        domain_rows = [
            (row, score)
            for row, score in zip(rows, output.scores, strict=True)
            if row["domain"] == domain
        ]
        metrics.setdefault(domain, {})["comet"] = round(
            statistics.fmean(score for _, score in domain_rows), 4
        )

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=0, help="0 = CPU, 1 = first GPU")
    args = parser.parse_args()

    print(f"loading {args.model}", file=sys.stderr)
    model = load_comet(args.model)
    rows: list[tuple[str, dict[str, Any], float | None]] = []
    for path in args.results:
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = score_file(model, path, args.batch_size, args.gpus)
        rows.append((data.get("model", path.name), metrics, data.get("rss_mb")))
        print(f"scored {path}", file=sys.stderr)

    print(
        f"{'model':<48} {'n':>5} {'chrF++':>8} {'BLEU':>8} {'COMET':>8} {'p50 ms':>8} {'RSS MB':>8}"
    )
    for name, metrics, rss in rows:
        latency = metrics.get("latency_ms", {}).get("p50", float("nan"))
        rss_value = rss if rss is not None else float("nan")
        print(
            f"{name[:48]:<48} {metrics.get('n', 0):>5} "
            f"{metrics.get('chrf++', float('nan')):>8.2f} "
            f"{metrics.get('bleu', float('nan')):>8.2f} "
            f"{metrics.get('comet', float('nan')):>8.4f} "
            f"{latency:>8.1f} {rss_value:>8.1f}"
        )


if __name__ == "__main__":
    main()
