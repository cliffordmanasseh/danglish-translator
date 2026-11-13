# Sermon domain specialization

This is the preaching-domain branch of the MT experiments. The goal is a
Danish→English student that preserves sermon style, biblical quotations, and
theological terminology better than the general OPUS-MT baseline.

## Data sources

- [Danske Prædikener](https://www.dansketaler.dk/praedikener) — 1,000+
  digitized Danish sermons from the Reformation to today. Each record keeps
  `source_url`, `source_note`, and `year`.
- Historical sermons (pre-1900) are public domain; the scraper defaults to
  `1850-1860` for that reason.
- Modern sermons (2025) are included for contemporary spoken style. Check the
  `source_note` on each record before redistributing the text; for an
  open-source dataset, prefer public-domain texts or texts with explicit
  permission.
- The Bible is a useful parallel source for quotations and terminology; use a
  public-domain Danish/English translation if you add it.

## Files

- `glossary.tsv` — Danish→English theological terms (synonyms separated by `|`).
- `scrape_sermons.py` — fetch sermon lists and pages, extract the body, keep
  source metadata; supports resumable `--links-only` / `--links` collection.
- `build_sermon_corpus.py` — normalize, deduplicate, and license-tag scraped
  sermons.
- `prepare_sermon_data.py` — split sermons into sentence-level sources; hold out
  entire sermons for the test split.
- `sermon_eval.py` — chrF++/BLEU/COMET plus glossary accuracy.
- `../generate_synthetic.py` — teacher generation; use `--system` and `--prompt`
  for the sermon prompt.
- `../train_student.py` — LoRA/full SFT on the resulting JSONL.
- `../mt_benchmark.py` — `--pairs` accepts a custom JSONL test set.

## Current corpus (2026-11-13)

- **1,305 sermons / 2.2M words**, from 1539 to 2026.
- **72,004 train sentences / 7,548 held-out test sentences**, split by whole
  sermon (`--test-every 10`).
- 163 pre-1900 sermons are `public_domain`; 1,142 modern sermons are
  `copyright_unknown` (local research only).
- Public-domain subset: **163 sermons / 600k words / 8,771 train sentences /
  2,356 test sentences** (`corpus_public_domain.jsonl`,
  `split_public_domain/`).
- Full corpus: `data/sermons/corpus.jsonl`; split: `data/sermons/split/`.
- Full report: `experiments/results/sermon/data-collection.md`.

## Bulk collection

The site's year filter is the allowed archive index; search pagination is
disallowed by robots.txt. Collect the link list first, then fetch the texts.
Both steps cache HTML and resume from existing output.

```bash
# 1. Collect links for a year range (the site rate-limits bursts; back off)
python experiments/sermon/scrape_sermons.py \
  --years 1900-2026 --links-only \
  --output data/sermons/dansketaler_links_modern.jsonl \
  --cache-dir data/sermons/cache --delay 3.0

# 2. Fetch the sermon texts from the link list
python experiments/sermon/scrape_sermons.py \
  --links data/sermons/dansketaler_links_modern.jsonl \
  --output data/sermons/dansketaler_sermons_modern.jsonl \
  --cache-dir data/sermons/cache --delay 1.0
```

The scraper accepts both `/tale/<slug>` and `/praedikener/tale/<slug>` URLs,
skips unsupported years that fall back to the current year, backs off on HTTP
429, and records `license` as `public_domain` for pre-1900 sermons and
`copyright_unknown` otherwise.

## Run

```bash
# 1. Scrape modern and historical sermons
python experiments/sermon/scrape_sermons.py \
  --years 2025 --limit 30 \
  --output /tmp/sermon/sermons-modern.jsonl

python experiments/sermon/scrape_sermons.py \
  --years 1850-1860 --limit 20 \
  --output /tmp/sermon/sermons-historical.jsonl

# 2. Sentence-level split, holding out entire sermons
python experiments/sermon/prepare_sermon_data.py \
  --input /tmp/sermon/sermons-modern.jsonl /tmp/sermon/sermons-historical.jsonl \
  --output-dir /tmp/sermon/data

# 3. Teacher-translate the train and test sources
python experiments/generate_synthetic.py \
  --input /tmp/sermon/data/sermon_train_sources.jsonl \
  --output /tmp/sermon/data/synthetic_train.jsonl \
  --url http://127.0.0.1:8731 --api openai \
  --model halogen-qwen3.8-flash-next --reasoning-effort none \
  --teacher Qwen3.8-Flash --workers 1 \
  --system "You are a professional Danish-to-English translator specializing in Christian sermons. Preserve biblical quotations, theological terminology, names, numbers, and references. Use a natural spoken preaching style. Output only the English translation." \
  --prompt "Translate the following Danish sermon excerpt into English. Danish: {source} English:"

# 4. Mix 500 sermon pairs + 500 general pairs
python experiments/prepare_pilot_data.py combine \
  --output-dir /tmp/sermon/data \
  --synthetic /tmp/sermon/data/synthetic_train.jsonl \
  --real /tmp/pilot2/pilot_real.jsonl \
  --max-synthetic 500 --max-real 500 --max-total 1000 \
  --output /tmp/sermon/data/sermon_train_mix.jsonl

# 5. LoRA fine-tune
python experiments/train_student.py \
  --model utter-project/EuroLLM-1.7B-Instruct \
  --data /tmp/sermon/data/sermon_train_mix.jsonl \
  --output /tmp/sermon/data/eurollm-1.7b-sermon-lora \
  --epochs 2 --batch-size 2 --grad-accum 8 --lr 1e-4 \
  --max-length 384 --device cpu --dtype float32 --merge

# 6. Build the test pairs (source + teacher reference)
python - <<'PY'
import json
from pathlib import Path
sources = {r["id"]: r for r in map(json.loads, Path("/tmp/sermon/data/sermon_test_sources.jsonl").read_text().splitlines())}
targets = {r["id"]: r for r in map(json.loads, Path("/tmp/sermon/data/synthetic_test.jsonl").read_text().splitlines())}
with Path("/tmp/sermon/data/sermon_test_pairs.jsonl").open("w") as out:
    for key, row in sources.items():
        if key in targets:
            out.write(json.dumps({"source": row["source"], "reference": targets[key]["target"], "domain": "sermon"}, ensure_ascii=False) + "\n")
PY

# 7. Evaluate
python experiments/mt_benchmark.py ct2 \
  --model-path models/opus-mt-tc-big-gmq-en-int8 \
  --pairs /tmp/sermon/data/sermon_test_pairs.jsonl \
  --limit-total 100 --output /tmp/sermon/data/opus-sermon-100.json

python experiments/mt_benchmark.py hf \
  --model-path /tmp/sermon/data/eurollm-1.7b-sermon-lora/merged \
  --pairs /tmp/sermon/data/sermon_test_pairs.jsonl \
  --limit-total 100 --max-tokens 128 \
  --device cpu --dtype float32 \
  --output /tmp/sermon/data/eurollm-sermon-100.json

python experiments/sermon/sermon_eval.py \
  --results /tmp/sermon/data/eurollm-sermon-100.json \
  --glossary experiments/sermon/glossary.tsv --comet
```

## Current status

- 50 sermons scraped: 30 modern (2025) + 20 historical (1850-1860; 8 rows were
  general `/tale/` links and are filtered by the scraper).
- 2,326 train sentences and 684 held-out test sentences (entire sermons held
  out).
- 200 test references generated with Qwen before the server stalled; the first
  100-pair test set is built.
- CC0 Bible seed: 62,113 verse pairs available; 1,000 sampled for training.
- Pilot training mix: 500 Bible + 500 general pairs.
- tc-big baseline on the 100 sermon pairs: **chrF++ 61.78, BLEU 43.88,
  COMET 0.8381, glossary 34/43 = 79.1%**.
- On the 30-pair subset used for the fast comparison:

| Model | chrF++ | BLEU | COMET | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big | 62.96 | 43.19 | **0.8530** | **6/7 = 85.7%** |
| EuroLLM-1.7B + Bible/general LoRA | **63.45** | **47.14** | 0.8492 | 5/7 = 71.4% |

The Bible proxy improved chrF++/BLEU but did **not** improve COMET and reduced
glossary accuracy on this small sample. It is not a win. The Bible's older
English is the likely problem: it teaches terminology but also archaic style.
The next run should use sermon-specific teacher pairs, not Bible verses.

### `agy` teacher check

`agy` (Gemini 3.8 Flash High) was evaluated on the same 30 sermon pairs before
committing to bulk generation:

| Model | chrF++ | BLEU | COMET | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big | 62.96 | 43.19 | 0.8530 | 6/7 = 85.7% |
| EuroLLM-1.7B + Bible LoRA | 63.45 | 47.14 | 0.8492 | 5/7 = 71.4% |
| `agy` Gemini 3.8 Flash High | **71.48** | **54.36** | **0.8876** | 6/7 = 85.7% |

`agy` is **+0.0346 COMET** over tc-big on this sample and keeps glossary
accuracy. That is a strong teacher signal. It is the teacher for the next
sermon run.

Bulk generation uses `generate_with_agy.py` in batches (10-25 items per call).
The agent CLI needs `--dangerously-skip-permissions` in headless mode; the
prompt explicitly says not to use tools.

The `agy` pilot used **200 `agy` sermon pairs + 500 general pairs** (700
examples) and a 1-epoch LoRA fine-tune. The 200 pairs were generated at
`gemini-3.8-flash-high`.

| Model (30 sermon pairs) | chrF++ | BLEU | COMET | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big | 62.96 | 43.19 | 0.8530 | **6/7 = 85.7%** |
| EuroLLM-1.7B + Bible LoRA | 63.45 | 47.14 | 0.8492 | 5/7 = 71.4% |
| EuroLLM-1.7B + `agy` LoRA (200 pairs) | **66.44** | **49.56** | **0.8600** | 5/7 = 71.4% |
| `agy` teacher | 71.48 | 54.36 | 0.8876 | 6/7 = 85.7% |

This is the first **win over tc-big on COMET** (+0.0070), plus +3.5 chrF++ and
+6.4 BLEU. The student retains only part of the teacher's +0.0346 gain, and
glossary accuracy is one term lower on a 30-pair sample with only 7
opportunities.

The original references are Qwen machine translations, so the student was also
scored against **independent editor references** (the same 30 sentences
translated from the Danish without seeing the machine output):

| Model (30 sermon pairs, editor refs) | chrF++ | BLEU | COMET | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big | 62.26 | 41.29 | 0.8339 | **6/7 = 85.7%** |
| EuroLLM-1.7B + `agy` LoRA (200 pairs) | **64.58** | **46.20** | **0.8422** | 5/7 = 71.4% |
| Qwen3.8 Flash (teacher) | 71.58 | 52.65 | 0.8835 | 6/7 = 85.7% |
| `agy` Gemini 3.8 Flash (teacher) | **75.09** | **60.11** | **0.8848** | 6/7 = 85.7% |
| `agy` Claude Opus 5.5 (teacher) | 73.87 | 58.88 | 0.8732 | 6/7 = 85.7% |

The win holds on references the student never saw (+0.0083 COMET, +2.3 chrF++,
+4.9 BLEU), so it is not an artifact of Qwen-generated references. Gemini also
leads Claude here (0.8848 vs 0.8732); the earlier Claude edge was noise on the
Qwen references.

### Qwen bulk generation

`agy` bulk generation was stopped to preserve the user's tokens. The 200 `agy`
pairs are kept, and Qwen3.8 Flash is the bulk teacher. On the independent
editor references Qwen is only **0.0013 COMET** behind Gemini (0.8835 vs
0.8848), which validates it as a bulk teacher.

The main 262k-context Qwen container wedges in kernel memory compaction and is
auto-restarted by the `halogen-qwen.service` user unit. Stop that service before
starting a dedicated container, or the 62 GB main model will refragment memory:

```bash
systemctl --user stop halogen-qwen.service
```

The dedicated server uses a 1024 context, one slot, a 1024-position KV pool, and
`HALOGEN_MAX_TOK=512`. Requests are at most ~170 prompt tokens, so that is
plenty. It runs at ~1.1 pairs/s, versus 0.34 pairs/s and a wedge after ~30
minutes with the 4096/4096 configuration:

```bash
podman run -d --name halogen-qwen-sermon3 \
  --network halogen-internal -p 127.0.0.1:8732:8731 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ipc=host --ulimit memlock=-1:-1 \
  -e HALOGEN_REASONING_EFFORT=low \
  -e HALOGEN_CTX=1024 -e HALOGEN_KV_SLOTS=1 \
  -e HALOGEN_MAX_TOK=512 -e HALOGEN_KV_POOL_POSITIONS=1024 \
  -e HALOGEN_PREFILL_CHUNK=512 \
  -v /data/halogen-models:/models:ro \
  ghcr.io/peonist-ai/halogen-flash-server:0.18.0
```

The 2,326-pair generation is running against port 8732. If it wedges, restart
the dedicated container and rerun the same command; `generate_synthetic.py`
resumes from the output ids. If the old engine is stuck exiting in kernel
`exit_mm`, `posix_fadvise(POSIX_FADV_DONTNEED)` on the model and n-gram files can
free ~38 GB of clean page cache and let it finish; the pinned weights are not
affected.

On the 200 sources translated by both teachers, Qwen and Gemini agree at
**chrF++ 77.8 / BLEU 63.8** with nearly identical length ratios (1.063 vs
1.056 source words). The bulk Qwen data is therefore stylistically compatible
with the 200 `agy` pairs.

The training mix is built with `combine_train_data.py`: all Qwen pairs + the
200 `agy` pairs (deduplicated by source, `agy` wins) + 500 general replay
pairs. Train for 1-2 epochs and evaluate on both the Qwen and editor reference
variants:

```bash
python experiments/train_student.py \
  --data /tmp/sermon/data/sermon_qwen_agy_mix.jsonl \
  --output /tmp/sermon/data/eurollm-1.7b-sermon-qwen-lora \
  --epochs 1 --batch-size 2 --grad-accum 8 --lr 1e-4 --merge
```

### ROCm training (gfx1151)

The halogen image already has ROCm 7.14 and the gfx1151 device package, so the
same image can train the student on the iGPU. Create a persistent training
container:

```bash
podman run -d --name halogen-train \
  --entrypoint sleep \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ipc=host --ulimit memlock=-1:-1 \
  -v /data/danglish_translator:/workspace:ro \
  -v /tmp/sermon/data:/data:rw \
  -v /data/rocm-target:/data/rocm-target:rw \
  -v /tmp/opencode/hf-cache:/hf-cache:rw \
  -e HF_HOME=/hf-cache -e PYTHONUNBUFFERED=1 \
  ghcr.io/peonist-ai/halogen-flash-server:0.18.0 infinity
```

Install the matched AMD PyTorch wheel, PEFT, and the missing system libraries:

```bash
podman exec halogen-train apt-get install -y libatomic1 libgomp1 libnuma1 gcc
podman exec halogen-train pip install --no-cache-dir \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'torch==2.13.0+rocm7.14.0' 'amd-torch-device-gfx1151==2.13.0+rocm7.14.0'
podman exec halogen-train pip install --no-cache-dir peft accelerate sacrebleu
```

The image's `rocm-sdk-libraries` wheel omits `libMIOpen.so.1`,
`librocsolver.so.0`, and several other libraries while its RECORD still lists
them. Install the wheel into the disk-backed target and symlink the missing
files:

```bash
podman exec halogen-train pip install --no-deps --target /data/rocm-target \
  --no-cache-dir --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  'rocm-sdk-libraries==7.14.0'
podman exec halogen-train python \
  /workspace/experiments/sermon/link_rocm_libs.py --target /data/rocm-target
```

Train on the GPU:

```bash
podman exec -e HF_HUB_OFFLINE=1 halogen-train python3 \
  /workspace/experiments/train_student.py \
  --data /data/sermon_qwen_agy_mix.jsonl \
  --output /data/eurollm-1.7b-sermon-qwen-rocm \
  --epochs 1 --batch-size 8 --grad-accum 2 --lr 1e-4 \
  --dtype bfloat16 --bf16 --device cuda \
  --attn-implementation eager --merge
```

**SDPA fails on gfx1151** with `hipErrorInvalidValue`; eager attention works in
both bf16 and fp32. The GPU run does ~800 tokens/s versus ~50 on CPU. The
merged model is saved in bf16. Copy `tokenizer.model` from the base cache into
the merged directory, because transformers 5.x saves only the fast tokenizer
and the host evaluator loads the slow `LlamaTokenizer`.

### Qwen+agy student result

The Qwen+agy student scores **0.8506 COMET** (1 epoch) and **0.8567** (2
epochs) on the 60-pair independent editor set, versus **0.8396** for the
previous `agy` student and **0.8316** for tc-big. The same-setting `agy`
control scores **0.8348**, so the gain comes from the added Qwen data, not from
bf16/eager training. The 2-epoch student is still 0.0313 COMET behind the Qwen
teacher.

The **full-corpus run** (75,407 pairs: 71,475 Qwen + 200 Gemini + 3,732 general,
1 epoch) scores **0.8696 COMET / 69.07 chrF++ / 49.69 BLEU / 14/17 glossary**
on the same set. That is **+0.0129** over the previous best and **+0.0380** over
tc-big, and it matches the teacher's glossary score.

For the edge, use **Q5_K_M** rather than Q4: `-c 768 -b 128 -ub 64 -np 1
-ctk q8_0 -ctv q8_0` gives **1.45 GB RSS** and **0.8663 COMET** on the 60-pair
set, only 0.0033 below bf16. Q4_K_M loses 0.0136 (0.8560 at 1.83 GB), so Q5 is
the better trade-off for this model.

## Notes

- The 30-pair test set has two reference variants: the original Qwen machine
  references and the independent editor references
  (`sermon_test_30_editor.jsonl`). Use the editor variant for model
  comparison; the Qwen variant is circular for a Qwen-trained student. The
  editor set was extended to **60 pairs** (`sermon_test_60_editor.jsonl`) for
  the larger comparison; its first 30 rows are the original 30-pair set.
- The teacher references are machine translations, not human references. The
  editor variant is a single independent translation, not a human-verified
  gold set. Commission multiple human references for a real release.
- `sermon_eval.py` reports glossary hit rate and worst terms. Treat it as a
  terminology regression check, not a theological correctness proof.
- The current Qwen server runs in the `halogen-qwen2` podman container. If the
  engine reports kernel memory compaction, free memory or restart the
  container; bulk generation can wedge it on a small host.
