#!/bin/bash
# Wait for the Qwen sermon generation to finish, then filter, combine, and train.
# Each step is logged to /tmp/post-gen-pipeline.log. Failures stop the pipeline
# so the logs can be inspected.
set -u
cd /data/danglish_translator

LOG=/tmp/post-gen-pipeline.log
OUTPUT=data/sermons/qwen_train_full.jsonl
CLEAN=data/sermons/qwen_train_clean.jsonl
FINAL=data/sermons/qwen_train_final.jsonl
MIX=data/sermons/sermon_qwen_agy_full_mix.jsonl
CONTAINER=halogen-train

echo "$(date -Is) waiting for generator" >> "$LOG"
while pgrep -f "generate_synthetic.py.*qwen_train_full" >/dev/null 2>&1; do
  sleep 30
done

COUNT=$(wc -l < "$OUTPUT" 2>/dev/null || echo 0)
echo "$(date -Is) generator stopped, rows=$COUNT" >> "$LOG"
tail -3 /tmp/qwen-sermon-full.log >> "$LOG"

if ! grep -q "finished .* rows" /tmp/qwen-sermon-full.log; then
  echo "$(date -Is) ERROR: no finish line in generator log" >> "$LOG"
  exit 1
fi

# 1. Final structural audit and filter.
.venv/bin/python experiments/sermon/audit_generated_pairs.py \
  --input "$OUTPUT" \
  --output data/sermons/audit_final.json \
  --flagged data/sermons/flagged_final.jsonl >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "audit failed" >> "$LOG"; exit 1; fi

.venv/bin/python experiments/sermon/filter_generated_pairs.py \
  --input "$OUTPUT" \
  --output "$CLEAN" \
  --rejected data/sermons/qwen_train_rejected.jsonl >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "structural filter failed" >> "$LOG"; exit 1; fi

# 2. Free the GPU, start the ROCm training container.
podman stop -t 5 halogen-qwen-sermon4 >> "$LOG" 2>&1
python3 /tmp/drop_model_cache.py >> "$LOG" 2>&1
podman start "$CONTAINER" >> "$LOG" 2>&1
sleep 5
podman exec "$CONTAINER" python3 -c "import torch; print('cuda', torch.cuda.is_available())" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "training container torch failed" >> "$LOG"; exit 1; fi

# 3. Full round-trip score on the GPU.
podman exec -e HF_HOME=/workspace/models/hf-cache-qe -e HF_HUB_OFFLINE=1 "$CONTAINER" \
  python3 /workspace/experiments/sermon/roundtrip_audit.py \
  --input /workspace/data/sermons/qwen_train_clean.jsonl \
  --metadata /workspace/data/sermons/split/sermon_train_sources.jsonl \
  --sample 0 --device cuda --batch-size 64 \
  --scores /data/roundtrip_full_scores.jsonl \
  --output /data/roundtrip_full.json >> "$LOG" 2>&1
RT_EXIT=$?
echo "$(date -Is) roundtrip exit=$RT_EXIT" >> "$LOG"

if [ "$RT_EXIT" -eq 0 ]; then
  .venv/bin/python experiments/sermon/filter_by_roundtrip.py \
    --input "$CLEAN" \
    --scores /tmp/sermon/data/roundtrip_full_scores.jsonl \
    --metadata data/sermons/split/sermon_train_sources.jsonl \
    --output "$FINAL" \
    --rejected data/sermons/qwen_train_roundtrip_rejected.jsonl \
    --min-score 25 >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then echo "roundtrip filter failed" >> "$LOG"; exit 1; fi
else
  echo "$(date -Is) roundtrip failed; using structural-clean file" >> "$LOG"
  cp "$CLEAN" "$FINAL"
fi

# 4. Combine with the Gemini pairs and general replay.
.venv/bin/python experiments/sermon/combine_train_data.py \
  --qwen "$FINAL" \
  --agy /tmp/sermon/data/agy_train.jsonl \
  --general data/sermons/general_replay.jsonl \
  --output "$MIX" >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "combine failed" >> "$LOG"; exit 1; fi

# 5. Train one epoch on the iGPU.
podman exec -e HF_HUB_OFFLINE=1 "$CONTAINER" \
  python3 /workspace/experiments/train_student.py \
  --model utter-project/EuroLLM-1.7B-Instruct \
  --data /workspace/data/sermons/sermon_qwen_agy_full_mix.jsonl \
  --output /data/eurollm-1.7b-sermon-qwen-full \
  --epochs 1 --batch-size 8 --grad-accum 2 --lr 1e-4 \
  --dtype bfloat16 --bf16 --device cuda \
  --attn-implementation eager --merge --logging-steps 50 >> "$LOG" 2>&1
TRAIN_EXIT=$?
echo "$(date -Is) training exit=$TRAIN_EXIT" >> "$LOG"

if [ "$TRAIN_EXIT" -eq 0 ]; then
  cp -r /tmp/sermon/data/eurollm-1.7b-sermon-qwen-full/merged \
    /data/danglish_translator/models/sermon/eurollm-1.7b-sermon-qwen-full-merged >> "$LOG" 2>&1
  echo "$(date -Is) model copied to models/sermon/eurollm-1.7b-sermon-qwen-full-merged" >> "$LOG"
fi

echo "$(date -Is) pipeline done" >> "$LOG"
