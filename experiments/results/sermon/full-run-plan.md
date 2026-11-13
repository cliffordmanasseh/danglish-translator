# Full sermon-student run, 2026-11-13

## Phase 1 — Generate the English side (running)

- Teacher: local Qwen3.8 Flash on `127.0.0.1:8732` (1024 context, 1 slot).
- Input: `data/sermons/split/sermon_train_sources.jsonl` (72,004 sentences).
- Output: `data/sermons/qwen_train_full.jsonl`.
- Rate: ~1.4–1.5 pairs/s → **13–15 hours**.
- Resumable: `generate_synthetic.py` skips ids already in the output.
- Watchdog: `experiments/sermon/watch_qwen_generation.sh` restarts the server
  if `/health` fails and restarts the generator if it exits early.
- Log: `/tmp/qwen-sermon-full.log`; watchdog: `/tmp/qwen-watchdog.log`.

## Phase 2 — Generate test references

After Phase 1, generate the 7,548 held-out test sentences:

```bash
python experiments/generate_synthetic.py \
  --input data/sermons/split/sermon_test_sources.jsonl \
  --output data/sermons/qwen_test_full.jsonl \
  --url http://127.0.0.1:8732 --api openai \
  --model halogen-qwen3.8-flash-next --reasoning-effort none \
  --teacher Qwen3.8-Flash --workers 1 --max-tokens 256 \
  --system 'You are a professional Danish-to-English translator specializing in Christian sermons. Preserve biblical quotations, theological terminology, names, numbers, and references. Use a natural spoken preaching style. Output only the English translation.' \
  --prompt 'Translate the following Danish sermon excerpt into English. Danish: {source} English:'
```

Expected: ~1.5 hours. These references are teacher-generated, so the 60-pair
independent editor set remains the unbiased check.

## Phase 3 — Filter and combine

1. Structural filter — drops fragments, headers, truncated targets, bad
   length ratios, and Danish residue:

```bash
python experiments/sermon/filter_generated_pairs.py \
  --input data/sermons/qwen_train_full.jsonl \
  --output data/sermons/qwen_train_clean.jsonl \
  --rejected data/sermons/qwen_train_rejected.jsonl
```

2. Round-trip score the full corpus (GPU in `halogen-train` after generation):

```bash
HF_HOME=/workspace/models/hf-cache-qe python3 \
  /workspace/experiments/sermon/roundtrip_audit.py \
  --input /workspace/data/sermons/qwen_train_clean.jsonl \
  --metadata /workspace/data/sermons/split/sermon_train_sources.jsonl \
  --sample 0 --device cuda --scores /data/roundtrip_full_scores.jsonl \
  --output /data/roundtrip_full.json
```

3. Drop rows below **25 chrF++**. On the 5k sample this removed ~4% of scored
   rows, concentrated in the 1570s–1890s and fragments; modern rows were mostly
   untouched.

```bash
python experiments/sermon/filter_by_roundtrip.py \
  --input data/sermons/qwen_train_clean.jsonl \
  --scores data/sermons/roundtrip_full_scores.jsonl \
  --metadata data/sermons/split/sermon_train_sources.jsonl \
  --output data/sermons/qwen_train_final.jsonl \
  --rejected data/sermons/qwen_train_roundtrip_rejected.jsonl \
  --min-score 25
```

4. Combine with the high-quality teacher pairs and general replay:

```bash
python experiments/sermon/combine_train_data.py \
  --qwen data/sermons/qwen_train_final.jsonl \
  --agy /tmp/sermon/data/agy_train.jsonl \
  --general data/sermons/general_replay.jsonl \
  --output data/sermons/sermon_qwen_agy_full_mix.jsonl
```

Expected mix: ~72k Qwen + 200 Gemini + 3,732 general (Tatoeba/NTREX), deduped
by source.

## Phase 4 — Train on the iGPU

Restart the ROCm training container (it was stopped to give Qwen the GPU):

```bash
podman start halogen-train
```

Train 1 epoch first; 72k pairs already give ~4,750 optimizer steps, 13× the
previous run:

```bash
podman exec -e HF_HUB_OFFLINE=1 halogen-train python3 \
  /workspace/experiments/train_student.py \
  --model utter-project/EuroLLM-1.7B-Instruct \
  --data /workspace/data/sermons/sermon_qwen_agy_full_mix.jsonl \
  --output /data/eurollm-1.7b-sermon-qwen-full \
  --epochs 1 --batch-size 8 --grad-accum 2 --lr 1e-4 \
  --dtype bfloat16 --bf16 --device cuda \
  --attn-implementation eager --merge
```

Expected: ~2–3 hours/epoch. Evaluate before deciding on a second epoch; with
10× more data than the 2,826-pair run, a second epoch may overfit.

## Phase 5 — Evaluate

- 60-pair independent editor references (unbiased, primary).
- A 1,000–2,000 sentence random sample of the 7,548 test pairs with Qwen
  references (larger, teacher-biased).
- Glossary accuracy.
- Compare against the previous best (0.8567 bf16 / 0.8552 Q4) and tc-big.

## Phase 6 — Edge artifact

- Quantize the best checkpoint to Q4_K_M.
- Run `llama-server -c 768 -b 128 -ub 64 -np 1 -ctk q8_0 -ctv q8_0`.
- Measure RSS, p50 latency, and COMET on the 60-pair editor set.
- Keep the 1-epoch checkpoint if it quantizes better than the 2-epoch one.

## Risks

- The 72k run is long; the watchdog handles server stalls, and the generator
  resumes from the output.
- Teacher errors are amplified at this scale. The quality filter and the
  independent editor set are the safeguards.
- The 2020s sermons dominate the corpus; historical texts (1539–1899) add
  orthographic variety but may produce noisier teacher output.
- The modern corpus is copyrighted. The shippable subset is the 163
  public-domain sermons; the full run is for research/domain adaptation.

## Actual outcome (2026-11-13)

- Generation: **71,965 accepted**, 39 rejected.
- Structural filter: **71,674 kept**, 291 rejected.
- The full GPU round-trip failed on the container side (transformers 5.x Marian
  tokenizer mapping and a gfx1151 Marian forward error). The pre-1900 subset
  was scored on the host CPU instead: 10,516 rows, **40.8 corpus chrF++**, and
  859 rows (8.2%) below 25.
- Current training mix (structural-clean): **75,407** = 71,475 Qwen + 200
  Gemini + 3,732 general.
- Filtered mix ready for a second stage: **74,550** = 70,618 Qwen + 200 Gemini
  + 3,732 general.
- Training: 1 epoch, **4,713 optimizer steps**, ~800 tokens/s at the start,
  loss ~1.0 by step 600.
