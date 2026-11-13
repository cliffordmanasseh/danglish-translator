#!/bin/bash
# Run the structural and round-trip audits at generation checkpoints.
set -u
cd /data/danglish_translator

INPUT=data/sermons/qwen_train_full.jsonl
METADATA=data/sermons/split/sermon_train_sources.jsonl
LOG=/tmp/periodic-audit.log

for TARGET in 20000 50000 72004; do
  while true; do
    DONE=$(wc -l < "$INPUT" 2>/dev/null || echo 0)
    [ "$DONE" -ge "$TARGET" ] && break
    sleep 300
  done
  echo "$(date -Is) audit at $TARGET (actual $DONE)" >> "$LOG"
  .venv/bin/python experiments/sermon/audit_generated_pairs.py \
    --input "$INPUT" \
    --output "data/sermons/audit_${TARGET}.json" \
    --flagged "data/sermons/flagged_${TARGET}.jsonl" >> "$LOG" 2>&1
  .venv/bin/python experiments/sermon/roundtrip_audit.py \
    --input "$INPUT" --metadata "$METADATA" --sample 500 \
    --output "data/sermons/roundtrip_${TARGET}.json" >> "$LOG" 2>&1
  echo "$(date -Is) audit $TARGET complete" >> "$LOG"
done
