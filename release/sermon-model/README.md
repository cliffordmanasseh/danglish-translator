# Sermon Danish→English — Q5_K_M

A 1.7B Danish-to-English translation model specialized for Christian preaching.
Recommended edge artifact: `eurollm-sermon-qwen-full-q5_k_m.gguf` (1.2 GB,
~1.29 GB RSS, ~270 ms p50 on the ROCm host CPU).

## Quick start

```bash
# 1. Verify the files
sha256sum -c SHA256SUMS

# 2. Start the local server
llama-server \
  -m eurollm-sermon-qwen-full-q5_k_m.gguf \
  --host 127.0.0.1 --port 18080 \
  -c 768 -b 128 -ub 64 -np 1 \
  -ctk q8_0 -ctv q8_0 -t 16

# 3. Translate
python translate.py "Gud er kærlighed."
```

The prompt format is ChatML; see `MODEL_CARD.md`. The model expects
sentence-sized Danish input (the training data was split into 4–45 word
sentences).

## Quality

On 60 held-out sermons with independent editor references:

| Model | COMET | chrF++ | BLEU | Glossary |
|---|---:|---:|---:|---:|
| **Q5_K_M** | **0.8663** | **70.76** | **51.45** | 14/17 |
| bf16 | 0.8696 | 69.07 | 49.69 | 14/17 |
| Q4_K_M | 0.8560 | 67.05 | 46.42 | 14/17 |
| OPUS-MT tc-big | 0.8316 | 63.34 | 44.42 | 13/17 |

## Licensing

The base model is Apache-2.0. The fine-tuned weights are **research use only**
for now: the training data includes 1,142 modern sermons with unknown rights.
Do not redistribute the weights as Apache-2.0 without a rights review. See
`MODEL_CARD.md` for details.
