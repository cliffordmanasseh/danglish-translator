#!/usr/bin/env python3
"""Round-trip audit: translate English targets back to Danish and compare.

A low chrF++ between the back-translation and the original Danish source flags
possible omissions or meaning drift. This is a proxy, not a semantic proof; use
it to rank and sample, not to auto-delete.

Usage:
    python experiments/sermon/roundtrip_audit.py \
        --input data/sermons/qwen_train_full.jsonl \
        --metadata data/sermons/split/sermon_train_sources.jsonl \
        --sample 500 --output data/sermons/qwen_train_full.roundtrip.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import sacrebleu


def sentence_chrf(hypothesis: str, reference: str) -> float:
    return sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2).score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--sample", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-en-da")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scores", type=Path, help="optional JSONL of per-row scores")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = {}
    if args.metadata:
        meta = {
            row["id"]: row
            for row in map(json.loads, args.metadata.read_text(encoding="utf-8").splitlines())
        }

    if args.sample and args.sample < len(rows):
        if meta:
            by_decade: dict[int, list[dict]] = defaultdict(list)
            for row in rows:
                year = int(meta.get(row.get("id", ""), {}).get("year", 0))
                by_decade[(year // 10) * 10].append(row)
            rng = random.Random(args.seed)
            per_decade = max(1, args.sample // max(1, len(by_decade)))
            sampled = []
            for decade in sorted(by_decade):
                sampled.extend(
                    rng.sample(by_decade[decade], min(per_decade, len(by_decade[decade])))
                )
            rows = sampled
        else:
            rows = random.Random(args.seed).sample(rows, args.sample)

    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(args.model)
    model = MarianMTModel.from_pretrained(args.model).to(args.device)
    model.eval()

    backtranslations: list[str] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        encoded = tokenizer(
            [row["target"] for row in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(args.device)
        generated = model.generate(**encoded, max_length=256, num_beams=1)
        backtranslations.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))

    sources = [row["source"] for row in rows]
    corpus_chrf = sacrebleu.corpus_chrf(backtranslations, [sources], word_order=2).score
    scores = [sentence_chrf(hyp, src) for hyp, src in zip(backtranslations, sources, strict=False)]

    by_decade_scores: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=False):
        year = int(meta.get(row.get("id", ""), {}).get("year", 0))
        by_decade_scores[f"{(year // 10) * 10}s" if year else "unknown"].append(score)

    ranked = sorted(zip(scores, rows, backtranslations, strict=False), key=lambda item: item[0])
    summary = {
        "rows": len(rows),
        "corpus_chrf++": round(corpus_chrf, 2),
        "mean_chrf++": round(statistics.mean(scores), 2),
        "median_chrf++": round(statistics.median(scores), 2),
        "p10_chrf++": round(sorted(scores)[len(scores) // 10], 2),
        "by_decade": {
            decade: {
                "n": len(values),
                "mean": round(statistics.mean(values), 2),
                "median": round(statistics.median(values), 2),
            }
            for decade, values in sorted(by_decade_scores.items())
        },
        "lowest": [
            {
                "score": round(score, 2),
                "source": row["source"][:200],
                "target": row["target"][:200],
                "backtranslation": back[:200],
            }
            for score, row, back in ranked[:10]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.scores:
        with args.scores.open("w", encoding="utf-8") as handle:
            for score, row, back in zip(scores, rows, backtranslations, strict=False):
                handle.write(
                    json.dumps(
                        {
                            "id": row.get("id"),
                            "score": round(score, 2),
                            "source": row["source"],
                            "target": row["target"],
                            "backtranslation": back,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(json.dumps({key: value for key, value in summary.items() if key != "lowest"}, indent=2))


if __name__ == "__main__":
    main()
