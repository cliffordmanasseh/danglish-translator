# Danish -> English student pilot

This directory is a research harness, not part of the runtime package. The goal
is to test whether a fine-tuned EuroLLM-1.7B can beat the existing OPUS-MT
tc-big model on the project's Danish -> English workload before committing to
large-scale synthetic data or vocabulary pruning.

The first zero-shot bake-off is in
[`results/2026-11-12-mt-baseline.md`](results/2026-11-12-mt-baseline.md).
Short version: Q8 EuroLLM-1.7B ties OPUS-MT tc-big on NTREX COMET but loses on
Tatoeba and is much larger/slower. Fine-tuning is the missing piece; pruning
comes after that.

## 0. Dependencies

```bash
# From the repo root.  torch/transformers come from the `convert` extra.
uv pip install --python .venv/bin/python -r experiments/requirements.txt
```

On the ROCm host, install the ROCm PyTorch wheel separately. The training and
`hf` evaluation scripts use `torch.cuda.is_available()`, which is true for ROCm.

## 1. Teacher options

### Option A: Qwen 3.8 Flash (already running here, ROCm)

On this machine ROCm lives inside the `halogen-qwen` podman container, not on
the host. It serves an OpenAI-compatible API:

```bash
podman ps
curl http://127.0.0.1:8731/health
curl http://127.0.0.1:8731/v1/models
```

The model id is `halogen-qwen3.8-flash-next`. It runs at roughly 50 tok/s on the
Radeon 8060S and supports `reasoning_effort: "none"`, which removes the thinking
tokens for translation. Use the OpenAI backend:

```bash
python experiments/generate_synthetic.py \
  --input /tmp/pilot/pilot_sources.jsonl \
  --output /tmp/pilot/synthetic.jsonl \
  --url http://127.0.0.1:8731 \
  --api openai \
  --model halogen-qwen3.8-flash-next \
  --reasoning-effort none \
  --workers 4 \
  --teacher Qwen3.8-Flash \
  --limit 12000
```

For evaluation, use `mt_benchmark.py openai` with the same URL/model. Qwen3 is
Apache-2.0; verify the license of the specific `halogen` checkpoint before
shipping synthetic data derived from it.

### Option B: EuroLLM-22B-Instruct (the original Apache-2.0 teacher)

Use a 22B GGUF, for example
[`bartowski/utter-project_EuroLLM-22B-Instruct-2512-GGUF`](https://huggingface.co/bartowski/utter-project_EuroLLM-22B-Instruct-2512-GGUF).
Q5_K_M is a good quality/size compromise. With a ROCm llama.cpp build:

```bash
llama-server \
  -m EuroLLM-22B-Instruct-2512-Q5_K_M.gguf \
  -ngl 99 \
  -c 4096 \
  -b 2048 \
  -ub 512 \
  -np 8 \
  --cont-batching \
  --host 127.0.0.1 --port 18082
```

`-np 8` must be at least the generator's `--workers` value. The host has no
ROCm user-space, so this option needs a ROCm container with llama.cpp (or a
Vulkan build). Option A is the faster path on this machine.

## 2. Fetch the evaluation data

The defaults expect the same files used for the baseline:

```bash
mkdir -p /tmp/opencode/mt-bench/data/tatoeba
curl -L -o /tmp/tatoeba.zip \
  https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/da-en.txt.zip
unzip -o /tmp/tatoeba.zip -d /tmp/opencode/mt-bench/data/tatoeba

curl -L -o /tmp/opencode/mt-bench/data/ntrex_da.txt \
  https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/newstest2019-ref.dan.txt
curl -L -o /tmp/opencode/mt-bench/data/ntrex_en.txt \
  https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/newstest2019-src.eng.txt
```

Tatoeba is CC-BY 2.0 FR; NTREX is CC-BY-SA 4.0. The benchmark uses 250
Tatoeba and 250 NTREX pairs sampled with seed 13.

## 3. Build disjoint source pools

```bash
python experiments/prepare_pilot_data.py sources \
  --output-dir /tmp/pilot \
  --synthetic-limit 12000 \
  --real-limit 12000
```

This writes:

- `pilot_sources.jsonl` — 12k Tatoeba Danish sources for synthetic generation.
- `pilot_real.jsonl` — 12k disjoint Tatoeba pairs plus the non-benchmark NTREX
  pairs, for the real-data half of the mixture.
- `pilot_manifest.json` — counts and seed.

The 250 Tatoeba + 250 NTREX benchmark sources (seed 13) are excluded, so the
pilot cannot train on the evaluation set.

## 4. Generate synthetic pairs

```bash
python experiments/generate_synthetic.py \
  --input /tmp/pilot/pilot_sources.jsonl \
  --output /tmp/pilot/synthetic.jsonl \
  --url http://127.0.0.1:18082 \
  --workers 8 \
  --teacher EuroLLM-22B-Instruct-2512 \
  --limit 12000
```

For the local Qwen3.8 Flash server, swap the backend flags:

```bash
python experiments/generate_synthetic.py \
  --input /tmp/pilot/pilot_sources.jsonl \
  --output /tmp/pilot/synthetic.jsonl \
  --url http://127.0.0.1:8731 \
  --api openai \
  --model halogen-qwen3.8-flash-next \
  --reasoning-effort none \
  --workers 1 \
  --teacher Qwen3.8-Flash \
  --limit 12000
```

On this machine, one worker was faster than four: the server's four slots
serialized the short requests and the health endpoint timed out under
contention. The sequential run did about 2.4-2.6 pairs/s; the four-worker run
did about 0.5 pairs/s.

Output is resumable JSONL. Rejected rows go to `synthetic.jsonl.rejects.jsonl`
with a reason (empty, copy, length ratio, special token). Inspect the rejects
before training; the length-ratio filter is deliberately loose.

At the measured ~2.5 pairs/s with Qwen3.8 Flash, 12k pairs is roughly 80
minutes. A slower 22B teacher should budget 2.5-5 hours. Run it overnight if
needed.

## 5. Mix synthetic and real data

```bash
python experiments/prepare_pilot_data.py combine \
  --output-dir /tmp/pilot
```

This writes `/tmp/pilot/pilot_train.jsonl` with `kind: synthetic|real`, one row
per source (synthetic wins on duplicates), shuffled with seed 42.

## 6. Fine-tune the student

LoRA first. The default is 1 epoch, effective batch 32, lr 1e-4, r=16, all
linear projections.

```bash
python experiments/train_student.py \
  --model utter-project/EuroLLM-1.7B-Instruct \
  --data /tmp/pilot/pilot_train.jsonl \
  --output /tmp/pilot/eurollm-1.7b-da-en-lora \
  --epochs 1 \
  --batch-size 4 \
  --grad-accum 8 \
  --lr 1e-4 \
  --bf16 \
  --gradient-checkpointing \
  --merge
```

The merged model is written to
`/tmp/pilot/eurollm-1.7b-da-en-lora/merged`. For a full fine-tune, add
`--full` and use a smaller learning rate (for example `2e-5`); the 128 GB
unified-memory host can fit it, but LoRA is the faster pilot.

## 7. Evaluate with the same 500 pairs

```bash
python experiments/mt_benchmark.py hf \
  --model-path /tmp/pilot/eurollm-1.7b-da-en-lora/merged \
  --name eurollm-1.7b-da-en-lora \
  --device cuda \
  --dtype bfloat16 \
  --limit 250 \
  --output experiments/results/eurollm-ft.json

python experiments/score_comet.py \
  experiments/results/opus-big.json \
  experiments/results/eurollm-ft.json
```

`score_comet.py` adds per-sentence and system COMET scores and rewrites the
JSONs. Use `--gpus 1` on the ROCm host if COMET can see the device.

If you prefer llama.cpp for the final edge check, convert the merged model to
GGUF with `llama.cpp/convert_hf_to_gguf.py`, quantize to Q4_K_M/Q5_K_M, and use
the `llm` subcommand instead.

## 8. Gate

Proceed to scale/prune only if the fine-tuned student:

- beats OPUS-MT tc-big by **>= 0.005 COMET overall**, and
- beats it by **>= 0.01 COMET on NTREX**.

If it does not, fine-tune `opus-mt-tc-big-gmq-en` on the same synthetic + real
mixture. That is the cheaper edge win and keeps the 229 MB / ~95 ms profile.

If it does pass, the next steps are:

1. Scale synthetic data to 100k-300k pairs with the 22B teacher.
2. Prune the 1.7B vocabulary to ~50k-64k (keep special tokens and byte
   coverage), then do a recovery fine-tune.
3. Quantize to Q4_K_M/Q5_K_M, measure RSS with `-c 1024 -b 512 -ub 256`, and
   benchmark on the actual edge device.

## Sermon domain branch

For the church-preaching use case, see [`sermon/README.md`](sermon/README.md).
It includes the sermon scraper, the theological glossary, the Bible CC0 seed
data, and the domain evaluation harness.

## Licensing notes

The model licenses are clean: EuroLLM-22B/1.7B, Qwen3, and OPUS-MT are
Apache-2.0. The data is the remaining audit surface. Tatoeba is CC-BY 2.0 FR;
NTREX is CC-BY-SA 4.0. For a commercial pipeline, prefer public-domain/CC0/
ODC-By Danish text or your own data, and keep the attribution notices. This is
not legal advice.

## Current status

The pipeline has been smoke-tested on the CPU-only development container:

- `prepare_pilot_data.py sources` works: 29,178 eligible Tatoeba pairs after
  filtering, with the 500 benchmark sources held out.
- `generate_synthetic.py` works end to end against a local llama.cpp server:
  5/5 sources accepted in the smoke run. The smoke run used the 1.7B Q4 as a
  stand-in; the 22B teacher was not available in this container, so the smoke
  output is not a teacher-quality measurement.
- `train_student.py` works with a tiny random Llama model: LoRA, optimizer,
  checkpointing, and merge all complete.
- `mt_benchmark.py hf` works with the tiny model.

The host has no ROCm user-space, but the `halogen-qwen` podman container does.
Qwen3.8 Flash is reachable on `127.0.0.1:8731` and was benchmarked as a teacher:
COMET **0.9190** vs tc-big **0.9117** on the 500-pair set. That is the strongest
teacher measured so far.

A 1,000-example pilot (500 Qwen-generated + 500 real) has finished. On a
20-pair check the student scored chrF++ 69.22 / COMET 0.9077 vs tc-big 68.19 /
0.9094: chrF++ improved, COMET was flat-to-slightly-down. The 20-pair sample is
noisy, but the 1k pilot is clearly too small to retain the teacher's +0.0073
COMET advantage. The next run needs the full 12k-pair mixture and more epochs,
on a ROCm PyTorch container rather than CPU. The CPU run managed ~45 tokens/s;
the 1.7B training is not practical to scale on CPU.
