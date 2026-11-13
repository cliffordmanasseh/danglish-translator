#!/usr/bin/env python3
"""Small, reproducible Danish -> English MT benchmark.

This is intentionally outside the ``danglish`` package: it is a research harness,
not a runtime dependency.  It supports two backend shapes:

* ``ct2``  -- a converted OPUS-MT / Marian directory (the runtime backend).
* ``llm``  -- a llama.cpp server exposing ``/completion``.

The default evaluation sets are Tatoeba da-en (everyday, conversational) and
NTREX-128 (news).  Both are small enough to run on CPU in minutes.

Examples
--------
    python experiments/mt_benchmark.py ct2 \
        --model-path models/opus-mt-da-en-int8 \
        --output experiments/results/opus-base.json

    python experiments/mt_benchmark.py llm \
        --url http://127.0.0.1:18080 \
        --server-pid 12345 \
        --output experiments/results/eurollm-q4.json

    python experiments/mt_benchmark.py score experiments/results/*.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_TATOEBA = Path("/tmp/opencode/mt-bench/data/tatoeba")
DEFAULT_NTREX = Path("/tmp/opencode/mt-bench/data")
DEFAULT_LIMIT = 250

# EuroLLM's model card uses this exact ChatML shape.  The trailing "English:"
# is part of the user turn; the assistant continues with the translation.
LLM_PROMPT = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\n"
    "Translate the following Danish source text to English:\n"
    "Danish: {source}\n"
    "English: <|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _clean(text: str) -> str:
    text = text.strip()
    # Some instruction-tuned checkpoints echo a language label.
    text = re.sub(r"^(Danish|English)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def load_tatoeba(directory: Path, limit: int, seed: int) -> list[dict[str, str]]:
    da_path = directory / "Tatoeba.da-en.da"
    en_path = directory / "Tatoeba.da-en.en"
    if not da_path.is_file() or not en_path.is_file():
        raise FileNotFoundError(f"Tatoeba files not found under {directory}")

    da = [line.strip() for line in da_path.read_text(encoding="utf-8").splitlines()]
    en = [line.strip() for line in en_path.read_text(encoding="utf-8").splitlines()]
    if len(da) != len(en):
        raise ValueError(f"Tatoeba files are not aligned: {len(da)} vs {len(en)}")

    pairs: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, reference in zip(da, en, strict=True):
        if not source or not reference or source in seen:
            continue
        if not (3 <= len(source.split()) <= 40 and 3 <= len(reference.split()) <= 50):
            continue
        seen.add(source)
        pairs.append({"source": source, "reference": reference, "domain": "tatoeba"})

    rng = random.Random(seed)
    if len(pairs) > limit:
        pairs = rng.sample(pairs, limit)
    return pairs


def load_ntrex(directory: Path, limit: int, seed: int) -> list[dict[str, str]]:
    da_path = directory / "ntrex_da.txt"
    en_path = directory / "ntrex_en.txt"
    if not da_path.is_file() or not en_path.is_file():
        raise FileNotFoundError(f"NTREX files not found under {directory}")

    da = [line.strip() for line in da_path.read_text(encoding="utf-8").splitlines()]
    en = [line.strip() for line in en_path.read_text(encoding="utf-8").splitlines()]
    if len(da) != len(en):
        raise ValueError(f"NTREX files are not aligned: {len(da)} vs {len(en)}")

    pairs: list[dict[str, str]] = []
    seen: set[str] = set()
    for source, reference in zip(da, en, strict=True):
        if not source or not reference or source in seen:
            continue
        if not (3 <= len(source.split()) <= 60 and 3 <= len(reference.split()) <= 80):
            continue
        seen.add(source)
        pairs.append({"source": source, "reference": reference, "domain": "ntrex"})

    rng = random.Random(seed)
    if len(pairs) > limit:
        pairs = rng.sample(pairs, limit)
    return pairs


def load_pairs(args: argparse.Namespace) -> list[dict[str, str]]:
    if getattr(args, "pairs", None):
        rows = [
            json.loads(line)
            for line in Path(args.pairs).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        pairs = [
            {
                "source": row["source"],
                "reference": row["reference"],
                "domain": row.get("domain", "custom"),
            }
            for row in rows
        ]
    else:
        tatoeba = load_tatoeba(Path(args.tatoeba), args.limit, args.seed)
        ntrex = load_ntrex(Path(args.ntrex), args.limit, args.seed)
        pairs = tatoeba + ntrex
    if args.limit_total:
        random.Random(args.seed).shuffle(pairs)
        pairs = pairs[: args.limit_total]
    return pairs


def rss_mb(pid: int | None = None) -> float | None:
    path = Path(f"/proc/{pid}/status") if pid else Path("/proc/self/status")
    try:
        for line in path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        return None
    return None


def llm_translate(url: str, source: str, max_tokens: int) -> tuple[str, float]:
    payload = {
        "prompt": LLM_PROMPT.format(source=source),
        "temperature": 0.0,
        "n_predict": max_tokens,
        "top_k": 1,
        "repeat_penalty": 1.0,
        "stop": ["<|im_end|>"],
        "stream": False,
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.load(response)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return _clean(body.get("content", "")), latency_ms


OPENAI_PROMPT = "Translate the following Danish source text to English:\nDanish: {source}\nEnglish:"


def openai_translate(
    url: str,
    model: str,
    source: str,
    *,
    max_tokens: int,
    reasoning_effort: str | None,
) -> tuple[str, float, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": OPENAI_PROMPT.format(source=source)}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.load(response)
    latency_ms = (time.perf_counter() - started) * 1000.0
    content = body["choices"][0]["message"].get("content") or ""
    return _clean(content), latency_ms, body


def run_ct2(args: argparse.Namespace, pairs: list[dict[str, str]]) -> dict[str, Any]:
    from danglish.backends.ct2 import CT2Backend

    backend = CT2Backend(
        args.model_path,
        beam_size=args.beam_size,
        intra_threads=args.threads,
    )
    backend.warmup()

    predictions: list[dict[str, Any]] = []
    for pair in pairs:
        output, latency_ms = "", 0.0
        started = time.perf_counter()
        output = backend.translate([pair["source"]], beam_size=args.beam_size)[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        predictions.append(
            {
                **pair,
                "prediction": _clean(output),
                "latency_ms": round(latency_ms, 2),
            }
        )

    return {
        "model": args.name or str(args.model_path),
        "backend": "ct2",
        "beam_size": args.beam_size,
        "threads": args.threads,
        "rss_mb": rss_mb(),
        "predictions": predictions,
    }


def run_llm(args: argparse.Namespace, pairs: list[dict[str, str]]) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    for pair in pairs:
        try:
            output, latency_ms = llm_translate(args.url, pair["source"], args.max_tokens)
        except urllib.error.URLError as exc:
            raise RuntimeError(f"llama.cpp server request failed: {exc}") from exc
        predictions.append(
            {
                **pair,
                "prediction": output,
                "latency_ms": round(latency_ms, 2),
            }
        )

    return {
        "model": args.name or args.url,
        "backend": "llm",
        "url": args.url,
        "max_tokens": args.max_tokens,
        "rss_mb": rss_mb(args.server_pid),
        "predictions": predictions,
    }


def run_openai(args: argparse.Namespace, pairs: list[dict[str, str]]) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    for pair in pairs:
        try:
            output, latency_ms, body = openai_translate(
                args.url,
                args.model,
                pair["source"],
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
            )
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible server request failed: {exc}") from exc
        timings = body.get("timings", {})
        predictions.append(
            {
                **pair,
                "prediction": output,
                "latency_ms": round(latency_ms, 2),
                "predicted_tokens": body.get("usage", {}).get("completion_tokens"),
                "predicted_per_second": timings.get("predicted_per_second"),
            }
        )

    return {
        "model": args.name or args.model,
        "backend": "openai",
        "url": args.url,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "rss_mb": rss_mb(args.server_pid) if args.server_pid else None,
        "predictions": predictions,
    }


def run_hf(args: argparse.Namespace, pairs: list[dict[str, str]]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA/ROCm is not available; falling back to CPU", file=sys.stderr)
        device = "cpu"

    if device == "cuda" and (
        args.dtype == "bfloat16" or (args.dtype == "auto" and torch.cuda.is_bf16_supported())
    ):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()

    predictions: list[dict[str, Any]] = []
    for pair in pairs:
        inputs = tokenizer(
            LLM_PROMPT.format(source=pair["source"]),
            return_tensors="pt",
        ).to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append(
            {
                **pair,
                "prediction": _clean(text),
                "latency_ms": round(latency_ms, 2),
            }
        )

    return {
        "model": args.name or args.model_path,
        "backend": "hf",
        "adapter": args.adapter,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "max_tokens": args.max_tokens,
        "rss_mb": rss_mb(),
        "predictions": predictions,
    }


def score_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    from sacrebleu.metrics import BLEU, CHRF

    hypotheses = [row["prediction"] for row in predictions]
    references = [[row["reference"] for row in predictions]]
    metrics: dict[str, Any] = {
        "n": len(predictions),
        "chrf++": round(CHRF(word_order=2).corpus_score(hypotheses, references).score, 2),
        "bleu": round(BLEU().corpus_score(hypotheses, references).score, 2),
    }
    for domain in sorted({row["domain"] for row in predictions}):
        rows = [row for row in predictions if row["domain"] == domain]
        metrics[domain] = {
            "n": len(rows),
            "chrf++": round(
                CHRF(word_order=2)
                .corpus_score(
                    [row["prediction"] for row in rows],
                    [[row["reference"] for row in rows]],
                )
                .score,
                2,
            ),
        }

    latencies = [row["latency_ms"] for row in predictions if row["latency_ms"] > 0]
    if latencies:
        ordered = sorted(latencies)
        metrics["latency_ms"] = {
            "mean": round(statistics.fmean(latencies), 1),
            "p50": round(ordered[int(0.50 * (len(ordered) - 1))], 1),
            "p95": round(ordered[int(0.95 * (len(ordered) - 1))], 1),
        }
    return metrics


def command_run(args: argparse.Namespace) -> None:
    pairs = load_pairs(args)
    print(f"loaded {len(pairs)} pairs", file=sys.stderr)
    if args.backend == "ct2":
        result = run_ct2(args, pairs)
    elif args.backend == "llm":
        result = run_llm(args, pairs)
    elif args.backend == "openai":
        result = run_openai(args, pairs)
    else:
        result = run_hf(args, pairs)
    result["metrics"] = score_predictions(result["predictions"])
    result["settings"] = {
        "limit_per_domain": args.limit,
        "seed": args.seed,
        "tatoeba": str(args.tatoeba),
        "ntrex": str(args.ntrex),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    print(f"wrote {output}", file=sys.stderr)


def command_score(args: argparse.Namespace) -> None:
    rows: list[tuple[str, dict[str, Any], float | None]] = []
    for path_str in args.results:
        path = Path(path_str)
        data = json.loads(path.read_text(encoding="utf-8"))
        old_metrics = data.get("metrics", {})
        metrics = score_predictions(data["predictions"])
        for domain, domain_metrics in old_metrics.items():
            if isinstance(domain_metrics, dict) and "comet" in domain_metrics:
                metrics.setdefault(domain, {})["comet"] = domain_metrics["comet"]
        comet_values = [
            row["comet"]
            for row in data["predictions"]
            if isinstance(row.get("comet"), (int, float))
        ]
        if comet_values:
            metrics["comet"] = round(statistics.fmean(comet_values), 4)
        data["metrics"] = metrics
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append((data.get("model", path.name), metrics, data.get("rss_mb")))

    print(
        f"{'model':<48} {'n':>5} {'chrF++':>8} {'BLEU':>8} {'COMET':>8} {'p50 ms':>8} {'RSS MB':>8}"
    )
    for name, metrics, rss in rows:
        latency = metrics.get("latency_ms", {}).get("p50", float("nan"))
        rss_value = rss if rss is not None else float("nan")
        comet = metrics.get("comet", float("nan"))
        print(
            f"{name[:48]:<48} {metrics['n']:>5} {metrics['chrf++']:>8.2f} "
            f"{metrics['bleu']:>8.2f} {comet:>8.4f} {latency:>8.1f} {rss_value:>8.1f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("ct2", "llm", "openai", "hf"):
        p = sub.add_parser(name)
        p.add_argument("--output", required=True)
        p.add_argument("--name")
        p.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="pairs per domain")
        p.add_argument("--limit-total", type=int, default=0)
        p.add_argument("--seed", type=int, default=13)
        p.add_argument("--tatoeba", default=str(DEFAULT_TATOEBA))
        p.add_argument("--ntrex", default=str(DEFAULT_NTREX))
        p.add_argument("--pairs", help="optional JSONL with source/reference/domain")
        p.add_argument("--max-tokens", type=int, default=256)
        p.add_argument("--server-pid", type=int)
        if name == "ct2":
            p.add_argument("--model-path", required=True)
            p.add_argument("--beam-size", type=int, default=2)
            p.add_argument("--threads", type=int, default=4)
        elif name == "llm":
            p.add_argument("--url", required=True)
        elif name == "openai":
            p.add_argument("--url", required=True)
            p.add_argument("--model", required=True)
            p.add_argument(
                "--reasoning-effort",
                default="none",
                help="set to 'none' to disable Qwen thinking tokens",
            )
        else:
            p.add_argument("--model-path", required=True)
            p.add_argument("--adapter")
            p.add_argument("--device", default="cuda")
            p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float32"])
            p.add_argument(
                "--attn-implementation",
                default="sdpa",
                choices=["eager", "sdpa", "flash_attention_2"],
                help="use eager on gfx1151 (SDPA fails with hipErrorInvalidValue)",
            )
        p.set_defaults(func=command_run, backend=name)

    p = sub.add_parser("score")
    p.add_argument("results", nargs="+")
    p.set_defaults(func=command_score)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
