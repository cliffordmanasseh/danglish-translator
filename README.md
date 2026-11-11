# Danglish

**Local, real-time Danish → English speech-to-speech translation under 2 GB of RAM.**

Speak Danish into your microphone and hear English from your speakers. Everything runs
on your CPU, on your machine: no cloud, no API keys, no telemetry. Danglish is designed
for small PCs, and the whole speech-to-speech stack fits in well under 2 GB of memory.

> **Status:** early but working. The text pipeline and the audio-to-audio path are both
> implemented and tested end to end. Capture, translation, and playback run in parallel;
> it is a turn-based interpreter (speak, pause, hear), not word-level simultaneous
> interpreting. `--full-duplex` keeps listening while the English audio plays.

## Demo

```bash
# 1. Install the speech stack, web demo, and one-time model converter
uv sync --extra speech --extra convert --extra server

# 2. Download and convert the translation model (uses torch once)
uv run --no-sync danglish fetch

# 3. Talk: Danish microphone in, English voice out
uv run --no-sync danglish talk

# With isolated input/output hardware (no acoustic feedback):
uv run --no-sync danglish talk --full-duplex
```

The first run downloads Whisper `small` (~464 MB), a Piper English voice (~63 MB), and
the OPUS-MT source model; the converted runtime model is ~77 MB. The one-time conversion
also installs torch and keeps the original model in the Hugging Face cache, so budget
roughly 1.5 GB of disk for setup. After that, model loading takes a few seconds. Use
headphones: the microphone is muted while English audio plays, which avoids a feedback
loop.

The converted `models/opus-mt-da-en-int8` directory is self-contained. If you want to run
inference on a machine that should not install torch, convert on a bigger machine and
copy that directory over.

No `uv`? With pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[speech,convert,server]" --extra-index-url https://download.pytorch.org/whl/cpu
danglish fetch
danglish talk
```

System dependency for audio on Debian/Ubuntu:

```bash
sudo apt install libportaudio2
```

### Text mode and web demo

The translator core also runs without audio:

```bash
uv run --no-sync danglish translate "God morgen, hvordan har du det?"
# Good morning, how are you?

uv run --no-sync danglish serve        # http://127.0.0.1:8765 (text UI, no microphone)
```

## How it works

```text
 microphone
     │
     ▼
┌───────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌──────────────┐
│ energy VAD│──▶│ Whisper small    │──▶│ OPUS-MT da→en      │──▶│ Piper TTS    │
│ 16 kHz    │   │ int8, Danish ASR │   │ CT2 int8, streaming│   │ English voice│
└───────────┘   └──────────────────┘   └────────────────────┘   └──────┬───────┘
                                                                      │
                                                                      ▼
                                                                  speaker
```

The microphone path is segmented by a small energy VAD. When you stop speaking for
600 ms, the utterance goes through local Whisper ASR, the sentence-aware translation
pipeline, and a local Piper voice. English audio starts playing as soon as the turn is
translated.

Capture/VAD, ASR→MT→TTS, and playback run on separate threads. By default the microphone
is muted during playback (half-duplex) to avoid feedback with laptop speakers. With
isolated audio I/O or an external echo canceller, pass `--full-duplex`: capture and
translation continue while English audio plays, and new turns queue behind the audio that
is currently playing. The models are still used by a single translation worker, so
memory does not increase; on a very small CPU, concurrent work can add a little latency.

The text path uses the same translation core and adds a debounced partial preview of the
sentence currently being written.

See [`docs/architecture.md`](docs/architecture.md) for the streaming algorithm, the
half-duplex audio model, and the memory strategy. Detailed measurements are in
[`docs/benchmarks.md`](docs/benchmarks.md).

### Why these models

- **Whisper small, int8** is the smallest Whisper size that is reasonably usable for
  Danish. It is ~464 MB on disk and runs at roughly 14× real time on the test machine.
- **Helsinki-NLP/opus-mt-da-en** is a 77M-parameter Marian model, converted to
  CTranslate2 int8 (~77 MB). It is a strong small general-purpose baseline, not a
  state-of-the-art translator.
- **Piper en_GB-alba-medium** is a 63 MB English voice that synthesizes at roughly 20×
  real time on CPU and is licensed CC BY 4.0.

## Measured footprint

Measured on a 32-core x86_64 development machine, Python 3.12, CTranslate2 int8,
4 inference threads, beam size 2. RSS is process resident memory; latency will vary by
CPU. Reproduce with `danglish bench` and `scripts/speech_smoke.py`.

| Configuration | Weights on disk | Peak RSS | Notes |
|---|---:|---:|---|
| Text translation | ~77 MB | **147 MB** | 10-sentence corpus, p50 36 ms |
| Text + web server | ~77 MB | **185 MB** | FastAPI/uvicorn after model load |
| Full speech-to-speech | ~604 MB | **~1.0 GB** | ASR small + MT + Piper, multi-turn |
| Hard budget | — | **2,048 MB** | target for the whole process |

Full-duplex uses the same models and bounded audio queues; its memory footprint is
unchanged. The extra cost is CPU contention with playback, which is small because
playback itself is cheap.

Measured turn latency for a 20-second Danish audio clip:

| Stage | Time | Real-time factor |
|---|---:|---:|
| Whisper small ASR | 1.3–1.4 s | ~14× real time |
| OPUS-MT translation | 0.12–0.13 s | — |
| Piper speech synthesis | 0.35–0.45 s | ~20–30× real time |
| **Total** | **~1.8–2.0 s** | for 20 s of input |

With the default 600 ms end-of-speech pause, expect roughly **1–2 seconds from “stop
speaking” to English audio** for short utterances, plus playback time. The first run
downloads models; subsequent process startups load them in a few seconds.

The 20-second clip above was one continuous utterance. When the VAD splits the same clip
into five turns, total processing is ~5.4 s (five Whisper calls, each with fixed
overhead) and peak RSS is ~1.0 GB. Short fragments also give Whisper less context, so
ASR quality drops; this is the normal trade-off of a turn-based interpreter. Tune
`--silence-ms` and `--min-energy` for your microphone and speaking style.

### Fitting under 2 GB

The full stack peaks at about 1.0 GB (1,024 MB observed after four passes over the
five-turn clip), leaving roughly 1 GB of headroom for the OS and other applications. If
you need to go lower:

- `danglish talk --asr-model base` lowers ASR memory but hurts Danish accuracy.
- Use `configs/small-pc.toml` (2 threads, greedy decoding) on a 2–4 core machine.
- The text-only pipeline is ~150 MB if you do not need audio.

`danglish doctor` reports an estimate against the 2 GB budget, and `danglish bench`
fails if measured peak RSS exceeds it. `danglish talk` prints the speech-stack RSS after
loading.

## CLI

```text
danglish talk                     microphone → Danish ASR → English speech (alias: mic)
danglish talk --full-duplex       keep listening while English audio plays (isolated I/O)
danglish talk --no-tts            microphone → Danish ASR → English text only
danglish talk --list-devices      list audio devices
danglish translate "text" | -f file | -        one-shot text translation
danglish repl                     interactive line-by-line text translation
danglish stream                   incremental stdin → stdout, with partials
danglish serve                    local text WebSocket demo + web UI
danglish fetch                    download + convert the OPUS-MT model
danglish doctor                   environment, dependency, and memory checks
danglish bench                    text latency and peak-RSS benchmark
```

Model and performance options live in [`configs/default.toml`](configs/default.toml).
Useful overrides:

```bash
danglish talk --asr-model small --tts-voice en_GB-cori-high
danglish talk --min-energy 0.008 --silence-ms 500
danglish talk --input-device 2 --output-device 3
danglish talk --full-duplex --input-device 2 --output-device 3
danglish talk --stats               # ASR/MT/TTS timings per turn
danglish talk --json                # machine-readable turns, still speaks
```

Every option can be overridden on the command line (`--model-path`, `--compute-type`,
`--threads`, `--beam-size`) or with `DANGLISH_MODEL` / `DANGLISH_CONFIG`.

## Speech quality and limitations

- **It is turn-based.** The VAD waits for 600 ms of silence before cutting the utterance,
  and Whisper transcribes the whole utterance. There is no partial ASR, so very long
  monologues wait for the full segment (max 15 s).
- **Half-duplex by default, full-duplex available.** The mic is muted during playback by
  default to protect laptop users. With isolated input/output, `--full-duplex` never
  mutes: capture and ASR/MT/TTS continue while the previous English audio plays.
  Playback is still serialized, so new translations queue behind the audio currently
  playing. This is parallel processing, not overlapping English speech.
- **Small-model quality.** Whisper `small` makes occasional Danish ASR mistakes, and
  OPUS-MT is a compact model. Proper names, numbers, and domain terms are the usual
  failure points. Check the printed Danish transcript when something sounds wrong.
- **One speaker, one stream.** The design is a local interpreter, not a multi-user
  meeting transcriber.

### Full-duplex with isolated audio

If your interface keeps the microphone and speaker signals separate, use:

```bash
danglish talk --full-duplex --input-device 2 --output-device 3 --stats
```

The capture callback keeps feeding the VAD while playback runs. A new Danish turn can be
recognized, translated, and synthesized while the previous English sentence is still
playing; its audio is queued in order. The translation worker is intentionally
single-threaded so turns do not reorder and only one copy of each model is loaded.

This is not yet word-level simultaneous interpreting: Whisper still waits for the VAD to
close an utterance, so 600 ms of silence (or a 15 s maximum segment) is required before a
new translation starts. Streaming ASR with partial hypotheses is the next latency step;
see the roadmap. On a 2–4 core machine, overlapping ASR with playback can add a little
latency, but memory stays flat because the models are shared.

## Repository layout

```text
src/danglish/
  segmenter.py         Danish-aware incremental sentence splitting
  pipeline.py          streaming translation events and document translation
  backends/            CTranslate2 backend + mock backend
  speech.py            ASR → MT → TTS orchestration (device-free, testable)
  tts.py               Piper voice loading, download, and synthesis
  audio/
    vad.py             energy VAD
    asr.py             faster-whisper wrapper
    runtime.py         parallel capture / translation / playback runtime
    mic.py             microphone capture, playback, talk loop
    util.py            resampling and WAV helpers
  server/              FastAPI WebSocket server + text UI
scripts/
  speech_smoke.py      WAV file → English WAV, no microphone needed
tests/                 unit tests + integration tests (marked `integration`)
```

## Models and licenses

Danglish code is **Apache-2.0**. Models and voices are downloaded separately and keep
their own licenses:

| Component | Model | License |
|---|---|---|
| Translation | `Helsinki-NLP/opus-mt-da-en` | Apache-2.0 |
| ASR | OpenAI Whisper `small` via faster-whisper | MIT |
| TTS voice | Piper `en_GB-alba-medium` | CC BY 4.0 (voice dataset) |
| TTS runtime | `piper-tts` (optional) | GPL-3.0-or-later |

The default Piper voice is downloaded at runtime, not bundled. `piper-tts` is an
optional GPL dependency installed only by the `speech`/`tts` extras; the Apache-2.0 core
does not require it. See [`NOTICE`](NOTICE) and [`docs/models.md`](docs/models.md) for
attribution, licensing details, and alternatives. The TTS voice choice is configurable;
if you switch voices, check that voice's license.

## Roadmap

- [x] Streaming text pipeline + Danish sentence segmentation
- [x] CT2 int8 backend and model conversion
- [x] WebSocket text demo
- [x] Local microphone → English speech pipeline (ASR + MT + TTS)
- [x] Parallel capture, translation, and playback
- [x] Full-duplex mode for isolated audio I/O
- [ ] Browser microphone + server-side audio-to-audio
- [ ] Streaming/partial ASR for lower latency
- [ ] Acoustic echo cancellation for shared speaker/microphone setups
- [ ] Quality evaluation harness (FLORES-200 / CoRal samples)
- [ ] Prequantized packaged ASR/TTS artifacts for faster first run
- [ ] Optional higher-quality English voices

## Contributing

Issues and pull requests are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md)
first. The short version: keep the runtime lean, add tests, and report memory numbers
with performance changes.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
