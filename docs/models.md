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
- NLLB-200-distilled-600M is stronger for multilingual use, but it needs a different
  tokenizer adapter and roughly 600 MB of int8 weights. It is not supported by the current
  backend without code changes.

## Speech recognition — Whisper small

| | |
|---|---|
| Source | OpenAI Whisper `small`, via [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) |
| Runtime | CTranslate2 int8, CPU |
| Disk | ~464 MB (downloaded by faster-whisper on first use) |
| Speed | ~14× real time on the test machine |
| License | MIT (Whisper weights and faster-whisper) |

`small` is the smallest Whisper size that is reasonably usable for Danish. `base` and
`tiny` are noticeably worse for Danish and are only recommended if memory is extremely
tight. The wrapper uses `language="da"`, `beam_size=1`, no internal VAD (the energy VAD
already segmented the audio), and no previous-text conditioning for short utterances.

Alternatives:

- Danish fine-tunes of Whisper (for example from the CoRal project) may improve
  conversational accuracy; check their size and license.
- Vosk has small Danish models, but their accuracy and timestamp behavior differ; an
  adapter would be a new backend.
- Whisper `medium` is better but roughly 1.5 GB of weights, which breaks the budget.

## Speech synthesis — Piper

| | |
|---|---|
| Runtime | [`piper-tts`](https://github.com/OHF-Voice/piper1-gpl) + ONNX Runtime |
| Runtime license | GPL-3.0-or-later (optional dependency) |
| Default voice | `en_GB-alba-medium` |
| Disk | 63 MB |
| Sample rate | 22,050 Hz mono |
| Speed | ~20–30× real time on the test machine |
| Voice license | CC BY 4.0 (dataset: DataShare Edinburgh) |

The voice is downloaded from [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices)
on first use and cached by Hugging Face. The voice is not bundled in this repository.

Recommended permissively licensed choices:

- `en_GB-alba-medium` — 63 MB, CC BY 4.0, British English, single speaker (default).
- `en_GB-cori-high` — 114 MB, public domain, higher quality.

Other Piper voices have their own licenses. Some, such as Lessac and Ryan, use
research/non-commercial dataset terms; do not assume a voice is redistributable just
because Piper can load it. Use `--tts-voice` or set `tts.voice` in the config. A local
voice can also be passed as a path to an `.onnx` file with a sibling `.onnx.json`.

`piper-tts` itself is GPL-3.0-or-later and is an optional dependency. The Apache-2.0
core does not require it; see [`../NOTICE`](../NOTICE) if you redistribute a combined
environment. A permissively licensed TTS backend would be a welcome contribution.

## Attribution and redistribution

Danglish code is Apache-2.0. Models and voices are separate works:

- keep the OPUS-MT model card and Apache-2.0 notice if you redistribute the converted
  model;
- Whisper and faster-whisper are MIT;
- the default Piper voice is CC BY 4.0 and requires attribution if redistributed.

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
