# Benchmarks

These numbers are from a development machine and are intended to show the shape of the
costs, not to promise a specific laptop's performance.

## Environment

- Ubuntu 26.04, x86_64, 32 logical cores, 122 GB RAM
- Python 3.12.14
- CTranslate2 4.9.0
- OPUS-MT da→en converted to int8, beam size 2
- faster-whisper 1.2.1, `lucasw89/whisper-small-danish-ct2-int8`, int8, 4 threads
- tiny-voca M2, ONNX Runtime + dizione, 2 threads
- optional Piper 1.8.0 `en_GB-alba-medium` fallback
- RSS measured with `danglish bench` / `scripts/speech_smoke.py` (process resident set)

## Text translation

Command:

```bash
danglish bench --repeats 3 --show-samples
```

Result:

```text
samples        30 translations (3x)
latency        p50 36.2 ms   p95 52.4 ms
throughput     283.5 target words/s
RSS after load 145.2 MB
peak RSS       146.8 MB
budget         2,048 MB — PASS (+1,901 MB headroom)
```

The corpus is ten original everyday Danish sentences in `src/danglish/corpus.py`.
Latency includes backend call overhead and SentencePiece encode/decode, but not model
load. The first sentence after startup is slower; `warmup()` removes most of that.

## Web server

The same text model behind FastAPI/uvicorn, measured with `/api/info` and one WebSocket
translation:

```text
RSS after startup: 159.6 MB
peak RSS:          184.8 MB (VmHWM)
```

The server adds roughly 40 MB over the text pipeline. The WebSocket round-trip for
“God morgen, hvordan har du det?” was 74 ms including the first-request overhead; a
preview returned 219 ms after the send, which is the 200 ms debounce plus 18 ms of
translation.

## Speech-to-speech

Input: a 20-second excerpt of the Danish spoken-Wikipedia article “Solen” from Wikimedia
Commons ([file page](https://commons.wikimedia.org/wiki/File:Da-Solen.ogg)), decoded to
16 kHz mono WAV. The audio is used locally for smoke testing and is not committed to the
repository; see the file page for its license and attribution.

Command:

```bash
uv run --no-sync python scripts/speech_smoke.py solen-20s.wav -o english.wav
```

Result (model files cached):

```text
DA  Dette er en optagelse af artiklen Solen fra Wikipedia. Det er versionen fra
    10. december 2008, indtalt den 11. december 2008 af bruger Daniel. Solen fra
    Wikipedia, den frie encyklopædi på da.wikia.dk
EN  This is a recording of the article The Sun from Wikipedia. It is the version from
    10 December 2008, recorded on 11 December 2008 by user Daniel. The sun from
    Wikipedia, the free encyclopedia of da.wikia.dk
wrote english.wav (15.67s)
asr 1.46s  mt 0.15s  tts 0.38s  wall 2.00s
RSS 891 MB  peak 891 MB
```

Switching from stock Whisper small to the Danish fine-tune changed the transcript from
“Diversionen … Danielen … encyklibedi” to “Det er versionen … bruger Daniel …
encyklopædi”. Translating the whole ASR turn instead of isolated sentences also fixed the
MT error “indtalt” → “recorded” (the sentence-level path produced “entered”).

A bigger MT model does not automatically help: on 1,000 OPUS-100 da→en test sentences,
the 77M model scored **BLEU 40.82 / chrF++ 59.58**, while
`opus-mt-tc-big-gmq-en` int8 scored **40.61 / 59.51** at roughly half the throughput. The
ASR was the real bottleneck for speech quality.

Peak RSS for the full speech stack was **~0.9 GB** for one long turn. During ASR model
load, CTranslate2 briefly allocates extra memory while quantizing the Whisper weights;
the peak is higher than the resident set immediately after loading.

### Multi-turn session

Running the same 20-second clip through the real energy VAD produced five turns
(3.54 s, 7.17 s, 1.05 s, 3.00 s, 2.70 s). Processing all five and synthesizing speech took
**5.38 s** wall time, with **900 MB RSS and 999 MB peak**. The fixed cost of each Whisper
call dominates short turns, but the Danish fine-tune handles the short fragments much
better than stock small.

```text
VAD utterances: 5 [3.54, 7.17, 1.05, 3.0, 2.7]
turn 0: audio 3.54s asr 1.07s mt 0.05s tts 0.17s
turn 1: audio 7.17s asr 0.98s mt 0.05s tts 0.20s
turn 2: audio 1.05s asr 0.80s mt 0.01s tts 0.06s
turn 3: audio 3.00s asr 0.89s mt 0.04s tts 0.13s
turn 4: audio 2.70s asr 0.82s mt 0.02s tts 0.09s
total wall 5.38s, RSS 900 MB, peak 999 MB
```

The short 1.05 s fragment is now “Solen.” instead of “Såt.”, and the next turn is “Fra
Wikipedia, den frie encyklopædi.” instead of a half-garbled phrase. The remaining
translation errors are MT-level, not ASR-level: e.g. “indtalt” becomes “entered” instead
of “recorded”. Use natural pauses and, if your microphone is noisy, raise
`--silence-ms` rather than cutting aggressively. The inner Silero VAD gate prevents
hallucinated words on silence and noise without trimming the utterance.

The parallel runtime (full-duplex by default) does not load extra models: capture, translation,
and playback overlap, but the same ASR/MT/TTS instances are shared and audio queues are
bounded. Peak RSS is therefore the same ~1.0 GB; the trade-off is CPU contention, which
is modest because playback is cheap. Re-measure on the target machine if you run ASR and
TTS concurrently on a 2-core CPU.

### Turning the numbers into perceived latency

For a 20-second utterance, processing takes ~2.0 seconds. For a short 3–5 second
utterance, ASR typically takes 0.8–1.1 seconds because of Whisper's fixed overhead. Add
the 300 ms VAD silence threshold and
the MT/TTS time; the typical “stop speaking → English audio” latency is roughly
**1–2 seconds**.

Model load after a warm cache, measured in a fresh process:

```text
ASR load   0.38 s   RSS 359 MB
TTS load   1.02 s   RSS 163 MB
TTS warmup 0.14 s
```

The first run additionally downloads ~238 MB (pre-quantized ASR), ~26 MB (tiny-voca
ONNX + phonemizer), and ~77 MB (MT), and performs a one-time CT2 conversion with torch.

## Caveats

- CPU frequency, core count, thermal throttling, and memory bandwidth all matter.
- ASR speed depends heavily on utterance length; Whisper processes 30-second windows.
- TTS speed depends on sentence length and whether ONNX Runtime has warmed up.
- RSS is process-level; shared libraries may be counted differently depending on how you
  measure. Use `danglish bench` and `scripts/speech_smoke.py` on your own machine.
