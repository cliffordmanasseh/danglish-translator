#!/bin/bash
# Generate English sermon pairs with the local Qwen teacher.
# Resumable: generate_synthetic.py skips ids already present in the output.
set -euo pipefail
cd /data/danglish_translator

exec .venv/bin/python experiments/generate_synthetic.py \
  --input data/sermons/split/sermon_train_sources.jsonl \
  --output data/sermons/qwen_train_full.jsonl \
  --url http://127.0.0.1:8732 --api openai \
  --model halogen-qwen3.8-flash-next --reasoning-effort none \
  --teacher Qwen3.8-Flash --workers 1 --max-tokens 256 \
  --system 'You are a professional Danish-to-English translator specializing in Christian sermons. Preserve biblical quotations, theological terminology, names, numbers, and references. Use a natural spoken preaching style. Output only the English translation.' \
  --prompt 'Translate the following Danish sermon excerpt into English. Danish: {source} English:'
