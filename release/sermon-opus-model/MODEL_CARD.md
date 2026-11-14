---
license: other
license_name: research-use-only
license_link: LICENSE-WEIGHTS
base_model: Helsinki-NLP/opus-mt-tc-big-gmq-en
language:
  - da
  - en
library_name: ctranslate2
tags:
  - translation
  - danish
  - english
  - sermons
  - opus-mt
  - ctranslate2
  - int8
---

# OPUS-MT tc-big Sermon Danish→English (int8)

A 234M-parameter Marian translation model fine-tuned for Christian preaching.
It preserves biblical quotations, theological terminology, names, numbers, and
references, and keeps the small/fast OPUS-MT deployment profile.

## Files

| File | Size |
|---|---:|
| `opus-mt-tc-big-gmq-en-sermon-int8/model.bin` | 225 MB |
| `opus-mt-tc-big-gmq-en-sermon-int8/source.spm` | 782 KB |
| `opus-mt-tc-big-gmq-en-sermon-int8/target.spm` | 778 KB |
| `opus-mt-tc-big-gmq-en-sermon-int8/shared_vocabulary.json` | 946 KB |

The whole CTranslate2 int8 folder is **228 MB**.

## Model

- **Base:** `Helsinki-NLP/opus-mt-tc-big-gmq-en` (Apache-2.0)
- **Architecture:** Marian, 234M parameters
- **Fine-tuning:** 1 epoch on 74,550 pairs, batch 32, lr 2e-5, bf16,
  eager attention (required on gfx1151)
- **Deployment:** CTranslate2 int8

A second epoch was tested and did **not** improve COMET (0.8654 vs 0.8658), so
the 1-epoch checkpoint is released.

## Training data

- 70,618 Qwen3.8 Flash sermon pairs
- 200 Gemini 3.8 Flash pairs
- 3,732 Tatoeba/NTREX general replay pairs
- Source corpus: 1,305 Danish sermons / 2.2M words, from
  [Danske Taler](https://www.dansketaler.dk/praedikener)
- 163 sermons are pre-1900 public domain; 1,142 modern sermons are
  `copyright_unknown`. The corpus is not distributed.

## Evaluation

199 held-out sentences from 115 sermons (entire sermons held out). References
are Qwen3.8 Flash translations; this favors models trained on Qwen data, so the
base-vs-fine-tuned comparison is the reliable signal.

| Model | COMET | chrF++ | BLEU | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big base | 0.8412 | 62.41 | 43.72 | 95/115 |
| **This model (1 epoch)** | **0.8658** | **68.92** | **52.15** | **97/115** |
| OPUS-MT, +2nd epoch | 0.8654 | 68.94 | 52.33 | 97/115 |
| EuroLLM-1.7B sermon, Q5_K_M | 0.8768 | 71.31 | 55.60 | 102/115 |

The fine-tune is **+0.0246 COMET** over the base and only **0.0110** behind the
1.7B EuroLLM, at ~1/6 the memory.

## Edge profile

Measured on the 32-core ROCm host, CTranslate2 int8, CPU:

| Metric | Value |
|---|---:|
| Disk | **228 MB** |
| RSS | **304 MB** |
| p50 latency | **143 ms** |
| Full speech stack | fits under 2 GB (ASR + MT + TTS ≈ 0.8 GB) |

## Usage

```bash
danglish translate \
  --model-path models/opus-mt-tc-big-gmq-en-sermon-int8 \
  "Gud er kærlighed."
```

or in `danglish.toml`:

```toml
[model]
path = "models/opus-mt-tc-big-gmq-en-sermon-int8"
compute_type = "int8"
```

The model uses the standard CTranslate2 Marian interface; the `danglish`
backend adds the source `</s>` token automatically.

## Limitations

- References are teacher-generated, not human gold.
- The evaluation is 199 sentences; differences below ~0.005 COMET are noise.
- Historical Danish (pre-1900) is the noisy tier.
- Glossary accuracy is 97/115, not perfect. Verify names, numbers, and biblical
  quotations for publication.
- This is not a Bible translation model.

## Licensing

- **Base model:** Apache-2.0 (`Helsinki-NLP/opus-mt-tc-big-gmq-en`).
- **Fine-tuned weights:** **research use only** — see
  [`LICENSE-WEIGHTS`](LICENSE-WEIGHTS). The training data includes 1,142 modern
  sermons with unknown rights, so the weights cannot be redistributed as
  Apache-2.0 without a rights review.
- The corpus and generated pairs are not included.

## Checksums

See `SHA256SUMS`.
