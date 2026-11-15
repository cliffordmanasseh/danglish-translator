# Models

Danglish is a pipeline of three independently replaceable models. This page documents the
defaults, their licenses, and the alternatives that fit a 2 GB CPU machine.

## Translation — OPUS-MT da→en

| | |
|---|---|
| Source | [`Helsinki-NLP/opus-mt-da-en`](https://huggingface.co/Helsinki-NLP/opus-mt-da-en) |
| Architecture | Marian encoder-decoder, 6+6 layers, 77M parameters |
| Runtime | CTranslate2 int8, CPU |
| Disk | ~77 MB (including SentencePiece models) |
| Peak RSS | ~147 MB for the text process |
| License | Apache-2.0 (per the model card) |
| Quality | Strong compact baseline for general text; Tatoeba chr-F 0.769 / BLEU 63.6 reported by the model card |

The model is trained on OPUS data and is good at everyday prose. It is not a
state-of-the-art translator: idioms, domain terminology, and long-range context are weak
spots. For a local 77 MB model, it is a very good starting point.

### Conversion

`danglish fetch` downloads the Hugging Face snapshot and runs
`ctranslate2.converters.TransformersConverter` with `quantization="int8"`. Conversion
needs `torch` and `transformers`, which are only in the `convert` extra; inference needs
neither.

**Important:** Marian's Hugging Face tokenizer appends the source `</s>` token. CTranslate2
does not do this automatically, and without it the model repeats itself until the
decoding limit. Danglish appends `</s>` explicitly. If you convert another OPUS-MT model,
keep this in mind.

### Alternatives

- Other OPUS-MT checkpoints (domain-specific or fine-tuned) can be converted with
  `danglish fetch --model-id ... --output ...` if they use the Marian + SentencePiece
  layout.
- Bigger OPUS-MT “tc-big” models can improve quality but add hundreds of megabytes and
  latency; check the converted int8 size against the budget.
- NLLB-200-distilled-600M is stronger for multilingual use, but it is CC-BY-NC-4.0
  (non-commercial), needs a different tokenizer adapter, and is roughly 600 MB of int8
  weights. It is not supported by the current backend without code changes.
- `Helsinki-NLP/opus-mt-tc-big-gmq-en` is a larger 6+6-layer Marian model (~228 MB int8).
  We benchmarked it on 1,000 OPUS-100 da→en sentences: BLEU 40.61 vs 40.82 for the 77M
  model, at roughly half the throughput. It is not a free quality win; try it on your own
  domain with `--model-path models/opus-mt-tc-big-gmq-en-int8` before adopting it.

## Speech recognition — Danish Whisper small

| | |
|---|---|
| Source | [`lucasw89/whisper-small-danish-ct2-int8`](https://huggingface.co/lucasw89/whisper-small-danish-ct2-int8), a CT2 conversion of [`thorhojhus/whisper-small-danish`](https://huggingface.co/thorhojhus/whisper-small-danish) |
| Base | OpenAI Whisper `small`, fine-tuned on Danish data (CoRal v3, Common Voice, FLEURS, FTSpeech, NST) |
| Runtime | CTranslate2 int8, CPU, via [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) |
| Disk | ~238 MB (pre-quantized int8) |
| Speed | ~14× real time on the test machine |
| License | MIT (weights and faster-whisper) |
| Accuracy | Model card: WER 0.078 vs 0.234 for stock small on a 21.5 s Danish clip |

This is the default. It has the same memory footprint and speed as stock Whisper `small`
but is much better on Danish. The wrapper uses `language="da"`, `beam_size=1`, no
previous-text conditioning, and **two VADs**: the cheap energy VAD in `audio/vad.py` cuts
utterances, and faster-whisper's inner Silero VAD removes silence/noise before decoding.
Without the inner VAD, the fine-tune can hallucinate a word such as “to” on silence or a
sine tone.

Our 20 s spoken-Wikipedia sample changed from:

```text
stock:  Det er en optagelse ... Diversionen ... indtaget ... brug af Danielen ... encyklibedi
danish: Dette er en optagelse ... Det er versionen ... indtalt ... bruger Daniel ... encyklopædi
```

Alternatives:

- Stock `small` remains available with `--asr-model small`; it is worse for Danish.
- Larger Danish fine-tunes (for example the CoRal `roest` models) may improve
  conversational accuracy but cost more memory; check licenses.
- `medium` / `large-v3-turbo` improve accuracy but push the full stack toward or beyond
  2 GB.
- Vosk has small Danish models, but their accuracy and timestamp behavior differ; an
  adapter would be a new backend.

## Speech synthesis — tiny-voca

| | |
|---|---|
| Source | [`n4ze3m/tiny-voca`](https://huggingface.co/n4ze3m/tiny-voca) |
| Architecture | 4.5M-parameter encoder, duration/prosody predictor, source-filter waveform head |
| Runtime | ONNX Runtime, CPU; [`dizione`](https://github.com/n4ze3m/dizione) phonemizer |
| Disk | 21 MB ONNX + 5 MB phonemizer data |
| Sample rate | 24,000 Hz mono |
| Speed | ~70–85× real time on 2–4 CPU threads |
| Voices | `F2` and `M2`, American English; `M2` is the default |
| License | Apache-2.0 (code and weights); CMUdict/homograph data licenses in the model repo |

The ONNX graph and phonemizer are downloaded from
[`n4ze3m/tiny-voca`](https://huggingface.co/n4ze3m/tiny-voca) on first use and cached by
Hugging Face. The Apache-2.0 model is trained from scratch on synthetic speech; it is
clear but can sound slightly synthetic, and the F2 voice sounds younger than intended.
It is English-only, has two fixed voices, and no streaming. Unusual names and spellings
are its main weakness.

In an informal 10-sentence sermon check, the M2 voice had no word errors when transcribed
by Whisper small; F2 read “Grace” as “Race” once. That is an intelligibility sanity check,
not a listening test.

### Piper fallback

Piper remains available for comparison and for British English voices:

| | |
|---|---|
| Runtime | [`piper-tts`](https://github.com/OHF-Voice/piper1-gpl) + ONNX Runtime |
| Runtime license | GPL-3.0-or-later (optional `tts-piper` extra) |
| Voice | `en_GB-alba-medium`, 63 MB, CC BY 4.0 |
| Sample rate | 22,050 Hz mono |
| Speed | ~20–30× real time on the test machine |

Install it with `uv sync --extra tts-piper`, then run
`danglish talk --tts-backend piper --tts-voice en_GB-alba-medium`. The optional
`en_GB-cori-high` voice is 114 MB and public domain. Other Piper voices have their own
licenses; some use research/non-commercial data terms, so do not assume a voice is
redistributable just because Piper can load it. A local voice can also be passed as a
path to an `.onnx` file with a sibling `.onnx.json`.

`piper-tts` is GPL-3.0-or-later and is not part of the default Apache-2.0 speech stack.
See [`../NOTICE`](../NOTICE) if you redistribute a combined environment.

## Attribution and redistribution

Danglish code is Apache-2.0. Models and voices are separate works:

- keep the OPUS-MT model card and Apache-2.0 notice if you redistribute the converted
  model;
- Whisper and faster-whisper are MIT;
- tiny-voca code and weights are Apache-2.0; its phonemizer data carries CMUdict and
  Wikipedia Homograph Data licenses, copied next to the downloaded data;
- the optional Piper voice is CC BY 4.0 and the Piper runtime is GPL-3.0-or-later.

See [`../NOTICE`](../NOTICE) for the summary. If you change the default voice, update the
notice and check the new voice's license.

## Evaluation

There is no baked-in BLEU number in the repository yet. A useful evaluation harness would:

1. Translate a held-out Danish set such as FLORES-200 `dan_Latn` with the same settings;
2. measure chrF/BLEU and inspect named entities and numbers;
3. run Danish conversational audio through ASR → MT and compare the final English text
   with a reference transcript/translation;
4. track peak RSS and p95 latency alongside quality, because improvements that exceed the
   memory budget are not acceptable for this project.

This is on the roadmap. Contributions with reproducible numbers are welcome.
