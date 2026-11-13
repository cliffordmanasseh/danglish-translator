#!/bin/bash
# Keep the Qwen teacher and the sermon-pair generator alive.
# Restarts the container if /health fails, and restarts the generator if it
# exits before the output reaches the target row count.
set -u
cd /data/danglish_translator

OUTPUT=data/sermons/qwen_train_full.jsonl
TARGET=72004
CONTAINER=halogen-qwen-sermon4
LOG=/tmp/qwen-watchdog.log

while true; do
  if ! curl -sf -m 10 http://127.0.0.1:8732/health >/dev/null 2>&1; then
    echo "$(date -Is) health failed; restarting $CONTAINER" >> "$LOG"
    podman restart "$CONTAINER" >> "$LOG" 2>&1
    sleep 60
  fi

  if ! pgrep -f "generate_synthetic.py.*qwen_train_full.jsonl" >/dev/null 2>&1; then
    DONE=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
    if [ "$DONE" -lt "$TARGET" ]; then
      echo "$(date -Is) generator stopped at $DONE/$TARGET; restarting" >> "$LOG"
      nohup bash experiments/sermon/run_qwen_generation.sh >> /tmp/qwen-sermon-full.log 2>&1 &
      sleep 30
    fi
  fi

  sleep 300
done
