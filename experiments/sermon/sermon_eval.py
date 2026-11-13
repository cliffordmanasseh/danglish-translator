#!/usr/bin/env python3
"""Evaluate sermon translations: standard MT metrics + glossary accuracy.

Example
-------
    python experiments/sermon/sermon_eval.py \
        --results experiments/results/eurollm-sermon-100.json \
        --glossary experiments/sermon/glossary.tsv \
        --comet
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_glossary(path: Path) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        danish = parts[0].strip()
        english = [part.strip().lower() for part in parts[1].split("|") if part.strip()]
        if danish and english:
            rows.append((danish, english))
    return rows


def term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def terminology_accuracy(
    predictions: list[dict[str, Any]],
    glossary: list[tuple[str, list[str]]],
) -> dict[str, Any]:
    opportunities = 0
    hits = 0
    per_term: Counter[str] = Counter()
    per_term_hits: Counter[str] = Counter()
    misses: list[dict[str, str]] = []

    # Longest terms first so multi-word terms win over their parts.
    ordered = sorted(glossary, key=lambda item: len(item[0]), reverse=True)

    for row in predictions:
        source = str(row["source"])
        target = str(row["prediction"]).lower()
        for danish, english in ordered:
            if not term_pattern(danish).search(source):
                continue
            opportunities += 1
            per_term[danish] += 1
            if any(term_pattern(term).search(target) for term in english):
                hits += 1
                per_term_hits[danish] += 1
            elif len(misses) < 40:
                misses.append(
                    {
                        "danish": danish,
                        "expected": "|".join(english),
                        "source": source[:240],
                        "prediction": str(row["prediction"])[:240],
                    }
                )

    accuracy = hits / opportunities if opportunities else 0.0
    worst = [
        {
            "term": term,
            "hits": per_term_hits[term],
            "opportunities": count,
            "accuracy": round(per_term_hits[term] / count, 3),
        }
        for term, count in per_term.most_common()
        if count >= 2
    ]
    worst.sort(key=lambda item: (item["accuracy"], -item["opportunities"]))
    return {
        "opportunities": opportunities,
        "hits": hits,
        "accuracy": round(accuracy, 4),
        "worst_terms": worst[:20],
        "misses": misses,
    }


def standard_metrics(predictions: list[dict[str, Any]], use_comet: bool) -> dict[str, Any]:
    from sacrebleu.metrics import BLEU, CHRF

    hypotheses = [str(row["prediction"]) for row in predictions]
    references = [[str(row["reference"]) for row in predictions]]
    metrics: dict[str, Any] = {
        "n": len(predictions),
        "chrf++": round(CHRF(word_order=2).corpus_score(hypotheses, references).score, 2),
        "bleu": round(BLEU().corpus_score(hypotheses, references).score, 2),
    }
    if use_comet:
        from comet import download_model, load_from_checkpoint

        model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        data = [
            {"src": row["source"], "mt": row["prediction"], "ref": row["reference"]}
            for row in predictions
        ]
        output = model.predict(data, batch_size=8, gpus=0, progress_bar=False)
        metrics["comet"] = round(float(output.system_score), 4)
        for row, score in zip(predictions, output.scores, strict=True):
            row["comet"] = round(float(score), 4)
    latencies = [row["latency_ms"] for row in predictions if row.get("latency_ms")]
    if latencies:
        metrics["p50_ms"] = round(statistics.median(latencies), 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--glossary", default=Path("experiments/sermon/glossary.tsv"), type=Path)
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    predictions = data["predictions"]
    glossary = load_glossary(args.glossary)
    metrics = standard_metrics(predictions, args.comet)
    terminology = terminology_accuracy(predictions, glossary)

    report = {
        "model": data.get("model", args.results.name),
        "metrics": metrics,
        "terminology": terminology,
    }
    if args.output:
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    print(f"model: {report['model']}")
    print(f"n: {metrics['n']}  chrF++: {metrics['chrf++']}  BLEU: {metrics['bleu']}", end="")
    if "comet" in metrics:
        print(f"  COMET: {metrics['comet']}", end="")
    print()
    print(
        f"terminology: {terminology['hits']}/{terminology['opportunities']} "
        f"= {terminology['accuracy']:.1%}"
    )
    print("worst terms:")
    for item in terminology["worst_terms"]:
        print(
            f"  {item['term']:<28} {item['hits']}/{item['opportunities']} ({item['accuracy']:.0%})"
        )


if __name__ == "__main__":
    main()
