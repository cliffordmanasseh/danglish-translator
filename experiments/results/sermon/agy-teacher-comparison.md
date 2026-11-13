# Sermon translation: teachers, students, and edge model

## Setup

- 30 held-out sermon sentences, extended to 60 for the final comparison
  (entire sermons are held out in both sets).
- Two reference sets:
  - **Qwen references**: the original test references, machine translations
    from Qwen3.8 Flash.
  - **Editor references**: the same sentences translated directly from the
    Danish without consulting the machine references
    (`experiments/sermon/build_editor_references.py`). This removes the
    circularity of scoring a Qwen-trained model against Qwen output.
- Metrics: chrF++, BLEU, COMET (`Unbabel/wmt22-comet-da`), and glossary
  accuracy. The 30-pair subset has 7 term opportunities; the 60-pair set has
  17.
- `agy` models were run through `experiments/sermon/generate_with_agy.py`,
  batch size 10, `--dangerously-skip-permissions` (headless mode requires it).

## Results

| Model | chrF++ | BLEU | COMET | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big | 62.96 | 43.19 | 0.8530 | 6/7 = 85.7% |
| EuroLLM-1.7B + Bible LoRA | 63.45 | 47.14 | 0.8492 | 5/7 = 71.4% |
| EuroLLM-1.7B + `agy` sermon LoRA (200 pairs) | 66.44 | 49.56 | 0.8600 | 5/7 = 71.4% |
| `agy` Gemini 3.8 Flash High (teacher) | 71.48 | 54.36 | 0.8876 | 6/7 = 85.7% |
| `agy` Claude Opus 5.5 High (teacher) | 70.76 | **55.93** | **0.8906** | 6/7 = 85.7% |

### Independent editor references

| Model | chrF++ | BLEU | COMET | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big | 62.26 | 41.29 | 0.8339 | 6/7 = 85.7% |
| EuroLLM-1.7B + Bible LoRA | 60.26 | 39.92 | 0.8330 | 5/7 = 71.4% |
| EuroLLM-1.7B + `agy` sermon LoRA (200 pairs) | **64.58** | **46.20** | **0.8422** | 5/7 = 71.4% |
| Qwen3.8 Flash (bulk teacher) | 71.58 | 52.65 | 0.8835 | 6/7 = 85.7% |
| `agy` Gemini 3.8 Flash High (teacher) | **75.09** | **60.11** | **0.8848** | 6/7 = 85.7% |
| `agy` Claude Opus 5.5 High (teacher) | 73.87 | 58.88 | 0.8732 | 6/7 = 85.7% |

The glossary score has one false positive: in *"må vi tro"*, `tro` is the verb
"believe", but `glossary.tsv` lists the noun "faith|belief". Every model that
translates the clause correctly is counted as a miss on that item, so the
glossary figures understate performance.

## Interpretation

- The original Qwen-reference ranking put Claude Opus 5.5 **+0.0030 COMET**
  ahead of Gemini 3.8 Flash. On the independent editor references the order
  reverses: Gemini leads by **+0.0116 COMET**, +1.2 chrF++, and +1.2 BLEU.
  The earlier Claude edge was reference bias on a 30-pair sample, not a real
  quality difference. **Gemini 3.8 Flash is the better `agy` teacher.**
- Qwen3.8 Flash is only **0.0013 COMET** behind Gemini on the independent
  references, and it is local and free. It is a sound bulk teacher. Its
  chrF++/BLEU are lower (71.58/52.65 vs 75.09/60.11), so keeping the 200
  Gemini pairs in the mix is worthwhile.
- The `agy`-trained 1.7B student beats tc-big by **+0.0083 COMET**, +2.3
  chrF++, and +4.9 BLEU on the independent references. The win is therefore
  **not** an artifact of Qwen-generated references; it holds on references the
  student never saw.
- The teacher gap is still large (0.8848 vs 0.8422 COMET), so more sermon data
  and more epochs should help.
- Glossary accuracy is 5/7 for the student vs 6/7 for tc-big and the teachers.
  One of the two "misses" is the `tro` verb/noun false positive above; the real
  miss is `skyld` translated as "blame" instead of "guilt/debt". Terminology is
  the remaining weakness, but it is one item on a 30-pair sample.

## Decision

**Stop bulk `agy` generation.** The token cost is the user's, and Gemini 3.8
Flash is the best `agy` teacher on independent references. Keep the **200
`agy` sermon pairs already generated**; they are enough for the pilot and add
teacher diversity to the Qwen bulk data.

For scaling beyond 200 pairs, use the local Qwen3.8 Flash server. The
independent-reference result validates Qwen as a bulk teacher: it is within
0.0013 COMET of Gemini and costs nothing.

No `generate_with_agy.py` process is currently running. The two remaining `agy`
processes on the host are the user's interactive sessions, not this batch job.

## Qwen bulk student (2,826 pairs)

Training mix: **2,126 Qwen + 200 `agy` + 500 general replay** examples.

### 30-pair independent editor references

| Model | chrF++ | BLEU | COMET | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big | 62.26 | 41.29 | 0.8339 | 6/7 = 85.7% |
| EuroLLM + `agy` LoRA (CPU float32, previous best) | 64.58 | 46.20 | 0.8422 | 5/7 = 71.4% |
| EuroLLM + `agy` LoRA (GPU bf16/eager control) | 64.31 | 45.32 | 0.8371 | 6/7 = 85.7% |
| EuroLLM + Qwen+`agy` LoRA (GPU bf16/eager) | **67.90** | **48.42** | **0.8515** | 5/7 = 71.4% |

The GPU control isolates the training backend. It is slightly **worse** than the
CPU float32 run (0.8371 vs 0.8422), so the new model's gain is not an artifact
of bf16/eager training. Against the same-setting agy control the Qwen mix is
**+0.0144 COMET**; against the previous best it is **+0.0093 COMET**, +3.3
chrF++, and +2.2 BLEU.

The student is still 0.033 COMET behind the Gemini teacher (0.8848), so there
is room to improve with more data or a second epoch.

### 60-pair independent editor references

| Model | chrF++ | BLEU | COMET | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big | 63.34 | 44.42 | 0.8316 | 13/17 = 76.5% |
| EuroLLM + `agy` LoRA (GPU bf16/eager control) | 64.77 | 45.84 | 0.8348 | 14/17 = 82.4% |
| EuroLLM + `agy` LoRA (CPU float32, previous best) | 64.37 | 45.78 | 0.8396 | 14/17 = 82.4% |
| EuroLLM + Qwen+`agy` LoRA, 1 epoch (bf16) | 67.29 | 46.72 | 0.8506 | 13/17 = 76.5% |
| EuroLLM + Qwen+`agy` LoRA, 2 epochs (bf16) | **68.09** | **47.92** | **0.8567** | 13/17 = 76.5% |
| Qwen3.8 Flash (teacher) | 73.31 | 54.43 | 0.8880 | 14/17 = 82.4% |

The larger sample confirms the result. The 1-epoch Qwen+agy student beats the
previous best by **+0.0110 COMET**, tc-big by **+0.0190**, and the same-setting
agy control by **+0.0158**. A second epoch adds another **+0.0061**, giving
**0.8567 COMET** (+0.0171 over the previous best, +0.0251 over tc-big). The
gain is from the Qwen data, not the GPU backend. The 2-epoch student is still
0.0313 COMET behind the Qwen teacher.

### Full-corpus student (75,407 pairs)

The full sermon corpus was generated with Qwen (71,965 accepted), structurally
filtered to 71,674, and combined with 200 Gemini + 3,732 general pairs.

| Model | COMET | chrF++ | BLEU | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big | 0.8316 | 63.34 | 44.42 | 13/17 = 76.5% |
| Previous best: 2.8k pairs, 2 epochs | 0.8567 | 68.09 | 47.92 | 13/17 = 76.5% |
| **Full 75k pairs, 1 epoch** | **0.8696** | **69.07** | **49.69** | **14/17 = 82.4%** |
| Qwen3.8 Flash (teacher) | 0.8880 | 73.31 | 54.43 | 14/17 = 82.4% |

The full-corpus student beats the previous best by **+0.0129 COMET**, tc-big by
**+0.0380**, and matches the teacher's glossary score. It is still 0.0184 COMET
behind the Qwen teacher. Training loss converged to ~0.89–0.93.

### Edge quantization

| Model | Quant | Context / KV | RSS after 60 req | p50 | COMET (60) |
|---|---|---:|---:|---:|---:|
| Full 75k, 1 epoch | bf16 HF | — | — | 460 ms* | **0.8696** |
| Full 75k, 1 epoch | **Q5_K_M** | 768 / q8_0 | **1.45 GB** | 644 ms* | **0.8663** |
| Full 75k, 1 epoch | Q4_K_M | 768 / q8_0 | 1.83 GB | 539 ms* | 0.8560 |
| Previous best 2.8k, 2 epochs | Q4_K_M | 768 / q8_0 | 1.83 GB | 259 ms | 0.8552 |

*Latency measured while the 2-epoch training was running; isolated latency will
be lower.

Q5_K_M is the best edge trade-off: it loses only **0.0033 COMET** from bf16 and
stays at **1.45 GB RSS**, while Q4 loses 0.0136. The full-corpus Q5 edge scores
**0.8663**, which is **+0.0096** over the previous best 2.8k Q4 (0.8552) and
**+0.0347** over tc-big (0.8316).

### ROCm training

Training now runs in the halogen container on the gfx1151 iGPU at **~800
tokens/s**, about 16× the CPU rate. SDPA fails on gfx1151 with
`hipErrorInvalidValue`; use `--attn-implementation eager`. The full container
setup is in `sermon/README.md`.
