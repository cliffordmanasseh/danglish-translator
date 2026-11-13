#!/usr/bin/env python3
"""Generate synthetic Danish -> English pairs with a llama.cpp teacher.

The teacher is expected to be served by ``llama-server`` with enough parallel
slots for ``--workers`` (for example ``-np 8 --cont-batching``).  The prompt is
the ChatML format used by the EuroLLM instruction models.

The output is JSONL and is resumable: rerunning with the same ``--output`` skips
source ids that are already present.  Rejected rows are written to
``<output>.rejects.jsonl`` with a reason, so filtering is inspectable.

Example
-------
    python experiments/generate_synthetic.py \
        --input /tmp/pilot/pilot_sources.jsonl \
        --output /tmp/pilot/synthetic.jsonl \
        --url http://127.0.0.1:18082 \
        --workers 8 \
        --limit 12000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROMPT = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\n"
    "Translate the following Danish source text to English:\n"
    "Danish: {source}\n"
    "English: <|im_end|>\n"
    "<|im_start|>assistant\n"
)

OPENAI_PROMPT = "Translate the following Danish source text to English:\nDanish: {source}\nEnglish:"

_LABEL_RE = re.compile(r"^(Danish|English)\s*:\s*", re.IGNORECASE)


def clean_translation(text: str) -> str:
    text = text.strip()
    text = text.replace("<|im_end|>", "").strip()
    text = _LABEL_RE.sub("", text)
    return text.strip()


def read_sources(path: Path) -> list[dict[str, Any]]:
    """Read JSONL (preferred) or a plain text file, one source per line."""

    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                source = str(obj.get("source", "")).strip()
                if not source:
                    continue
                rows.append(
                    {
                        "id": str(obj.get("id", len(rows))),
                        "source": source,
                        "domain": str(obj.get("domain", "unknown")),
                    }
                )
    else:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                source = line.strip()
                if source:
                    rows.append({"id": str(len(rows)), "source": source, "domain": "unknown"})
    return rows


def read_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                done.add(str(json.loads(line)["id"]))
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    return done


def request_translation_llama(
    url: str,
    source: str,
    *,
    prompt_template: str,
    system: str,
    max_tokens: int,
    token_ratio: float,
    timeout: float,
) -> dict[str, Any]:
    predicted = min(max_tokens, max(64, int(len(source.split()) * token_ratio)))
    raw_prompt = (prompt_template or PROMPT).format(source=source)
    if system:
        raw_prompt = f"<|im_start|>system\n{system}<|im_end|>\n{raw_prompt}"
    payload = {
        "prompt": raw_prompt,
        "temperature": 0.0,
        "n_predict": predicted,
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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "target": clean_translation(body.get("content", "")),
        "latency_ms": round(latency_ms, 1),
        "completion_tokens": int(body.get("tokens_predicted", 0)),
        "prompt_tokens": int(body.get("tokens_evaluated", 0)),
        "stopped": bool(body.get("stop", False)),
    }


def request_translation_openai(
    url: str,
    source: str,
    *,
    model: str,
    reasoning_effort: str | None,
    prompt_template: str,
    system: str,
    max_tokens: int,
    token_ratio: float,
    timeout: float,
) -> dict[str, Any]:
    predicted = min(max_tokens, max(64, int(len(source.split()) * token_ratio)))
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append(
        {
            "role": "user",
            "content": (prompt_template or OPENAI_PROMPT).format(source=source),
        }
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": predicted,
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
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    latency_ms = (time.perf_counter() - started) * 1000.0
    choice = body["choices"][0]
    usage = body.get("usage", {})
    timings = body.get("timings", {})
    return {
        "target": clean_translation(choice["message"].get("content") or ""),
        "latency_ms": round(latency_ms, 1),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "stopped": choice.get("finish_reason") == "stop",
        "predicted_per_second": timings.get("predicted_per_second"),
    }


def request_translation(
    url: str,
    source: str,
    *,
    api: str,
    model: str,
    reasoning_effort: str | None,
    prompt_template: str,
    system: str,
    max_tokens: int,
    token_ratio: float,
    timeout: float,
) -> dict[str, Any]:
    if api == "openai":
        return request_translation_openai(
            url,
            source,
            model=model,
            reasoning_effort=reasoning_effort,
            prompt_template=prompt_template,
            system=system,
            max_tokens=max_tokens,
            token_ratio=token_ratio,
            timeout=timeout,
        )
    return request_translation_llama(
        url,
        source,
        prompt_template=prompt_template,
        system=system,
        max_tokens=max_tokens,
        token_ratio=token_ratio,
        timeout=timeout,
    )


def rejection_reason(row: dict[str, Any], args: argparse.Namespace) -> str | None:
    source = row["source"]
    target = row["target"]
    if not target:
        return "empty"
    if target == source:
        return "copy"
    if "<|im_start|>" in target or "<|im_end|>" in target:
        return "special_token"
    source_words = len(source.split())
    target_words = len(target.split())
    if source_words == 0 or target_words == 0:
        return "empty_words"
    ratio = target_words / source_words
    if ratio < args.min_ratio:
        return f"too_short:{ratio:.2f}"
    if ratio > args.max_ratio:
        return f"too_long:{ratio:.2f}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--url", required=True)
    parser.add_argument("--api", choices=["llama", "openai"], default="llama")
    parser.add_argument("--model", default="halogen-qwen3.8-flash-next")
    parser.add_argument(
        "--reasoning-effort",
        default="none",
        help="OpenAI-compatible only; 'none' disables Qwen thinking tokens",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="max sources to process")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--token-ratio", type=float, default=2.5)
    parser.add_argument("--min-ratio", type=float, default=0.45)
    parser.add_argument("--max-ratio", type=float, default=2.6)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--teacher", default="teacher")
    parser.add_argument("--system", default="", help="optional system message")
    parser.add_argument(
        "--prompt",
        default="",
        help="optional prompt template with a {source} placeholder",
    )
    parser.add_argument("--flush-every", type=int, default=100)
    args = parser.parse_args()

    sources = read_sources(args.input)
    if args.limit:
        sources = sources[: args.limit]
    done = read_done_ids(args.output)
    pending = [row for row in sources if row["id"] not in done]
    print(
        f"{len(sources)} sources, {len(done)} already done, {len(pending)} pending",
        file=sys.stderr,
    )
    if not pending:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = args.output.with_suffix(args.output.suffix + ".rejects.jsonl")

    write_lock = threading.Lock()
    out_handle = args.output.open("a", encoding="utf-8")
    reject_handle = rejects_path.open("a", encoding="utf-8")
    started = time.perf_counter()
    completed = 0
    accepted = 0
    rejected = 0
    failures = 0

    def handle(row: dict[str, Any]) -> None:
        nonlocal completed, accepted, rejected, failures
        try:
            result = request_translation(
                args.url,
                row["source"],
                api=args.api,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                prompt_template=args.prompt,
                system=args.system,
                max_tokens=args.max_tokens,
                token_ratio=args.token_ratio,
                timeout=args.timeout,
            )
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            failures += 1
            record = {**row, "error": str(exc)}
            with write_lock:
                reject_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                reject_handle.flush()
            return

        record = {
            **row,
            "target": result["target"],
            "teacher": args.teacher,
            "backend": args.api,
            "prompt_version": (
                "custom"
                if args.prompt
                else ("chatml-da-en-v1" if args.api == "llama" else "da-en-openai-v1")
            ),
            "latency_ms": result["latency_ms"],
            "completion_tokens": result["completion_tokens"],
            "prompt_tokens": result["prompt_tokens"],
            "stopped": result["stopped"],
        }
        reason = rejection_reason(record, args)
        with write_lock:
            if reason is None:
                out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1
            else:
                reject_handle.write(
                    json.dumps({**record, "reject_reason": reason}, ensure_ascii=False) + "\n"
                )
                rejected += 1
            completed += 1
            if completed % args.flush_every == 0:
                out_handle.flush()
                reject_handle.flush()
                elapsed = time.perf_counter() - started
                rate = completed / elapsed if elapsed else 0.0
                print(
                    f"{completed}/{len(pending)} done  accepted={accepted} "
                    f"rejected={rejected} failed={failures}  {rate:.2f} pairs/s",
                    file=sys.stderr,
                )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(handle, row) for row in pending]
            for future in as_completed(futures):
                future.result()
    finally:
        out_handle.flush()
        reject_handle.flush()
        out_handle.close()
        reject_handle.close()

    elapsed = time.perf_counter() - started
    print(
        f"finished {completed} rows in {elapsed / 60:.1f} min: "
        f"accepted={accepted} rejected={rejected} failed={failures}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
