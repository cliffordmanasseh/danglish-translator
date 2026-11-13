# Danish -> English MT baseline, 2026-11-12

## Setup

- **Data:** 250 Tatoeba da-en pairs (everyday/conversational) + 250 NTREX-128
  pairs (news), sampled with seed 13. 500 pairs total.
- **Metrics:** chrF++ and BLEU via SacreBLEU; COMET via
  `Unbabel/wmt22-comet-da`.
- **OPUS-MT:** CTranslate2 int8, beam size 2, 4 intra-op threads.
- **EuroLLM:** `utter-project/EuroLLM-1.7B-Instruct` GGUF (Q4_K_M and Q8_0),
  llama.cpp CPU server, 4k context, greedy decoding, ChatML prompt from the
  model card.
- **Hardware:** 32-core AMD Ryzen AI MAX+ 395, 122 GB RAM. The container used for
  this run exposes no ROCm, so EuroLLM latency is CPU-only and is **not**
  representative of the ROCm host.

## Results

| Model | chrF++ | BLEU | COMET | NTREX chrF++ / COMET | Tatoeba chrF++ / COMET | p50 latency | RSS |
|---|---:|---:|---:|---|---|---:|---:|
| OPUS-MT da-en int8 (77 MB) | 63.19 | 42.48 | 0.9036 | 59.95 / 0.8666 | 76.44 / 0.9406 | 36 ms | 167 MB |
| OPUS-MT tc-big gmq-en int8 (229 MB) | **65.36** | **45.49** | **0.9117** | 62.27 / **0.8792** | **77.93** / **0.9442** | 95 ms | 319 MB |
| EuroLLM-1.7B-Instruct Q4_K_M (1.05 GB) | 63.71 | 41.92 | 0.8986 | 61.48 / 0.8733 | 72.74 / 0.9239 | 626 ms | 4.5 GB* |
| EuroLLM-1.7B-Instruct Q8_0 (1.76 GB) | 64.72 | 44.01 | 0.9038 | **62.38** / 0.8790 | 74.18 / 0.9286 | 423 ms | 4.5 GB* |
| Qwen3.8 Flash (ROCm container) | **65.58** | 44.06 | **0.9190** | **62.71** / **0.8904** | 77.16 / **0.9476** | 437 ms | 27 GB GTT† |

\* llama.cpp server `VmRSS` after 500 requests with `-c 4096 -b 2048 -ub 512`.
This is not a tuned edge configuration. The model file itself is 1.05 GB (Q4) /
1.76 GB (Q8). A separate Q4 run with `-c 2048 -b 512 -ub 256 -np 4` measured
**~1.84 GB RSS**, so the context/batch settings matter more than the Q4/Q8
weight difference for memory.

† Qwen3.8 Flash runs in the `halogen-qwen` podman container on gfx1151; the
server holds ~27 GB GTT. The p50 is the host-to-container HTTP round-trip. The
model is a custom `halogen-qwen3.8-flash-next` checkpoint; verify its license
before shipping synthetic data derived from it.

## Teacher check: Qwen3.8 Flash

The ROCm container is real and usable: `rocminfo` inside `halogen-qwen` reports
`gfx1151`, and the OpenAI-compatible server answers on
`http://127.0.0.1:8731`. On the same 500 pairs it has the best COMET of the
models tested:

- **+0.0073 COMET overall** over OPUS-MT tc-big (0.9190 vs 0.9117)
- **+0.0112 COMET on NTREX** (0.8904 vs 0.8792)
- ~50 tok/s generation; `reasoning_effort: "none"` removes thinking tokens

That makes it the practical teacher for the pilot. EuroLLM-22B remains the
clean EuroLLM alternative, but Qwen3.8 Flash is already running on this machine
and clears the teacher-quality gate.

## What the numbers say

1. **Zero-shot EuroLLM-1.7B is not yet a win.** Q8 is essentially tied with
   tc-big on NTREX COMET (0.8790 vs 0.8792) but loses on Tatoeba (0.9286 vs
   0.9442). Overall it is 0.008 COMET below tc-big. Q4 loses another ~0.005
   COMET and is 1.3 points below tc-big overall.
2. **Quantization costs are real but modest.** Q4 vs Q8: -0.005 COMET,
   -1.0 chrF++. That is acceptable for an edge model, but Q4 is starting from
   below tc-big, so it cannot absorb both the quantization loss and the
   fine-tuning gap.
3. **COMET is the fairer metric for the LLM.** The LLM often produces valid
   paraphrases ("My head still hurts" vs "I still have a headache") that chrF
   penalizes heavily. On COMET the gap is much smaller. This is consistent with
   the literature: small decoder LLMs can be competitive on semantic quality,
   especially with context.
4. **Tatoeba likely contaminates the OPUS-MT comparison.** Tatoeba is part of
   OPUS training data; OPUS-MT reproduces many references exactly. NTREX is the
   cleaner held-out set, and there Q8 ties tc-big on COMET. That is the most
   encouraging signal for distillation.
5. **The current 1.7B deployment is still tight for the 2 GB target.** With
   `-c 2048 -b 512 -ub 256 -np 4`, the Q4 server RSS was ~1.84 GB. Add Whisper
   small (~464 MB) and Piper (~63 MB) and the full stack is around 2.4 GB, over
   budget. The ROCm host will improve speed, not memory. A pruned 1.7B or a
   0.6B student is still the safer edge shape.

## Decision

Do **not** prune or quantize further as the next step. The unpruned Q8 student
is already close to tc-big on the cleaner benchmark; the missing piece is
task/domain fine-tuning, not vocabulary compression. Pruning now would stack a
recovery-fine-tuning requirement on top of an unproven student.

## Next experiment (pilot)

1. Generate 20k-50k synthetic da-en pairs with EuroLLM-22B-Instruct-2512 from
   permissive Danish text (plus 20k real parallel pairs).
2. LoRA fine-tune the **unpruned** EuroLLM-1.7B-Instruct on that pilot set.
3. Re-run this exact 500-pair benchmark.
4. Gate: beat tc-big by >= 0.005 COMET overall and >= 0.01 COMET on NTREX.
   If it passes, scale data and then prune/quantize. If it fails, fine-tune
   OPUS-MT tc-big on the same data; that is the cheaper edge win.

## Pilot pipeline

The pilot scripts are in `experiments/`:

- `prepare_pilot_data.py` — disjoint synthetic/real source pools, excluding the
  500 benchmark sentences.
- `generate_synthetic.py` — resumable 22B teacher generation via llama.cpp.
- `train_student.py` — LoRA/full SFT for EuroLLM-1.7B.
- `mt_benchmark.py hf` — evaluate a merged or adapter model with Transformers.

See `experiments/README.md` for the commands and gates.

## Pilot result

- ROCm is available inside the `halogen-qwen` podman container (gfx1151,
  ROCm runtime 1.21). The host itself has no ROCm user-space.
- Qwen3.8 Flash was benchmarked as a teacher: COMET 0.9190 vs tc-big 0.9117.
- 500 synthetic pairs were generated at ~2.5 pairs/s with zero rejections.
- The pilot mixes those 500 synthetic pairs with 500 real Tatoeba/NTREX pairs.
- EuroLLM-1.7B LoRA trained on CPU: 1,000 examples, 1 epoch, 63 optimizer
  steps. Final training loss was noisy around 3.5-4.6.
- The 100-pair evaluation was too slow on CPU, so the comparison was run on a
  smaller 20-pair sample (10 Tatoeba + 10 NTREX).

| Model (20 pairs) | chrF++ | BLEU | COMET |
|---|---:|---:|---:|
| OPUS-MT tc-big | 68.19 | 46.13 | **0.9094** |
| EuroLLM-1.7B LoRA (1k) | **69.22** | **46.47** | 0.9077 |

The student improved chrF++ by ~1 point and NTREX chrF++ by ~1.8 points, but
COMET is flat-to-slightly-down (-0.0017). The 20-pair sample is noisy, but this
is **not** a win yet. The 1k-example pilot is too small: the teacher has
+0.0073 COMET of headroom, and the student did not retain it in 63 CPU steps.
The next run needs the full 12k-pair mixture and more epochs, on a ROCm
training container rather than CPU.

## Sermon branch

The preaching-domain branch is in `experiments/sermon/`. First 30-pair check:

| Model | chrF++ | BLEU | COMET | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big | 62.96 | 43.19 | 0.8530 | **6/7 = 85.7%** |
| EuroLLM-1.7B + Bible/general LoRA | 63.45 | 47.14 | 0.8492 | 5/7 = 71.4% |
| EuroLLM-1.7B + `agy` sermon LoRA | **66.44** | **49.56** | **0.8600** | 5/7 = 71.4% |
| `agy` Gemini 3.8 Flash High (teacher) | 71.48 | 54.36 | 0.8876 | 6/7 = 85.7% |

The Bible proxy did not improve COMET. The `agy`-generated sermon data did:
**+0.0070 COMET over tc-big**, +3.5 chrF++, +6.4 BLEU. The student retains part
of the teacher's +0.0346 gain; the next run should use the full 2,326 sermon
pairs and 2-3 epochs.

## Reproduce

```bash
# OPUS-MT
python experiments/mt_benchmark.py ct2 \
  --model-path models/opus-mt-da-en-int8 \
  --name opus-mt-da-en-int8 \
  --limit 250 \
  --output experiments/results/opus-base.json

# EuroLLM (start llama.cpp server first)
python experiments/mt_benchmark.py llm \
  --url http://127.0.0.1:18080 \
  --server-pid <pid> \
  --name EuroLLM-1.7B-Instruct-Q4_K_M \
  --limit 250 \
  --output experiments/results/eurollm-q4.json
```

Raw prediction JSONs are in `experiments/results/` (copied from the test
machine). The harness lives at `experiments/mt_benchmark.py`.
