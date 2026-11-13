#!/usr/bin/env python3
"""Translate Danish to English with the sermon model's llama.cpp server.

Start the server first (see README.md), then:

    python translate.py "Gud er kærlighed."
    python translate.py --text "At de sidste skal blive de første."

No third-party dependencies; uses urllib.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PROMPT = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\n"
    "Translate the following Danish source text to English:\n"
    "Danish: {source}\n"
    "English: <|im_end|>\n"
    "<|im_start|>assistant\n"
)


def translate(server: str, source: str, max_tokens: int) -> str:
    body = {
        "prompt": PROMPT.format(source=source),
        "n_predict": max_tokens,
        "temperature": 0,
        "stop": ["<|im_end|>"],
    }
    request = urllib.request.Request(
        f"{server.rstrip('/')}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read())
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"could not reach {server}: {exc}\n"
            "Start the server with the command in README.md and try again."
        ) from exc
    return str(data.get("content", "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", help="Danish text to translate")
    parser.add_argument("--text", dest="text_option", help="Danish text to translate")
    parser.add_argument("--server", default="http://127.0.0.1:18080")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    source = args.text_option or args.text
    if not source:
        parser.error("provide Danish text as a positional argument or with --text")
    print(translate(args.server, source, args.max_tokens))


if __name__ == "__main__":
    main()
