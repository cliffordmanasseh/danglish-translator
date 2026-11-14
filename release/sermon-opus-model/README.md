# OPUS-MT sermon Danish→English — int8

A 228 MB CTranslate2 model for Danish-to-English preaching translation.
It runs in **304 MB RSS** at **143 ms p50** and scores **0.8658 COMET** on a
199-sentence held-out set — **+0.0246** over the OPUS-MT base and only
0.0110 behind the 1.7B EuroLLM.

## Quick start

```bash
# Extract the archive (files land in the current directory)
tar -xzf sermon-opus-int8.tar.gz

# Verify
sha256sum -c SHA256SUMS

# Translate through the Danglish app
danglish translate \
  --model-path ./opus-mt-tc-big-gmq-en-sermon-int8 \
  "Gud er kærlighed."
```

Or use it directly with CTranslate2:

```python
import ctranslate2
import sentencepiece as spm

model_dir = "opus-mt-tc-big-gmq-en-sermon-int8"
translator = ctranslate2.Translator(model_dir, device="cpu", compute_type="int8")
source = spm.SentencePieceProcessor(model_file=f"{model_dir}/source.spm")
target = spm.SentencePieceProcessor(model_file=f"{model_dir}/target.spm")

tokens = source.encode("Gud er kærlighed.", out_type=str) + ["</s>"]
result = translator.translate_batch([tokens], beam_size=2)[0]
print(target.decode(result.hypotheses[0]))
```

## Quality

| Model | COMET | chrF++ | BLEU | Glossary |
|---|---:|---:|---:|---:|
| OPUS-MT tc-big base | 0.8412 | 62.41 | 43.72 | 95/115 |
| **This model** | **0.8658** | **68.92** | **52.15** | **97/115** |
| EuroLLM-1.7B Q5 | 0.8768 | 71.31 | 55.60 | 102/115 |

199 held-out sentences from 115 sermons; references are Qwen-generated.

## Licensing

The base model is Apache-2.0. The fine-tuned weights are **research use only**
under [`LICENSE-WEIGHTS`](LICENSE-WEIGHTS): the training data includes 1,142
modern sermons with unknown rights. See `MODEL_CARD.md`.
