# Benchmarks

These numbers are from a development machine and are intended to show the shape of the
costs, not to promise a specific laptop's performance.

## Environment

- Ubuntu 26.04, x86_64, 32 logical cores, 122 GB RAM
- Python 3.12.14
- CTranslate2 4.9.0
- OPUS-MT da→en converted to int8, beam size 2
- faster-whisper 1.2.1, Whisper `small`, int8, 4 threads
- piper-tts 1.8.0, `en_GB-alba-medium`, ONNX Runtime
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
DA  Det er en optagelse af artiklen Solen fra Wikipedia. ...
EN  It is a recording of the article Sun from Wikipedia.
EN  The version from 10 December 2008, captured on 11 December 2008, is the use of Danielen.
EN  The sun from Wikipedia, the free encyclibedi at da.wikipedia.com
wrote english.wav (15.53s)
asr 1.30s  mt 0.13s  tts 0.36s  wall 1.80s
RSS 874 MB  peak 874 MB
```

The ASR transcript itself contains errors (“Diversionen”, “encyklibedi”), which the
translation preserves. This is the expected failure mode of a small ASR model on proper
nouns and rare words; it is not an MT bug.

A separate run that also back-transcribed the synthesized English with Whisper confirmed
that the audio output is intelligible:

```text
EN  It is a recording of the article Sun from Wikipedia.
EN  The version from 10 December 2008, captured on 11 December 2008, is the use of Danielen.
EN  The sun from Wikipedia, the free encyclibedi at da.wikipedia.com
```

Peak RSS for the full speech stack was **~0.9 GB** (874–880 MB) for one long turn,
depending on allocator behavior. During ASR model load, CTranslate2 briefly allocates
extra memory while quantizing the Whisper weights; the peak is higher than the resident
set immediately after loading.

### Multi-turn session

Running the same 20-second clip through the real energy VAD produced five turns
(3.54 s, 7.17 s, 1.05 s, 3.00 s, 2.70 s). Processing all five and synthesizing speech took
**5.38 s** wall time, with **882 MB RSS and 979 MB peak**. Repeating the five turns four
times stabilized at **925 MB RSS and 1,024 MB peak**, so the process does not grow without
bound during a conversation. The fixed cost of each Whisper call dominates short turns,
and the short 1.05 s fragment was transcribed poorly (“Såt.”). This is expected: give the
VAD natural sentence/paragraph boundaries rather than tuning it to cut every brief pause.

```text
VAD utterances: 5 [3.54, 7.17, 1.05, 3.0, 2.7]
turn 0: audio 3.54s asr 0.98s mt 0.05s tts 0.15s
turn 1: audio 7.17s asr 1.00s mt 0.06s tts 0.30s
turn 2: audio 1.05s asr 0.83s mt 0.02s tts 0.06s
turn 3: audio 3.00s asr 0.86s mt 0.04s tts 0.15s
turn 4: audio 2.70s asr 0.80s mt 0.02s tts 0.09s
total wall 5.38s, RSS 882 MB, peak 979 MB
```

The ASR transcript degrades on the short segments too (“Soen”, “Brua Daniele”), which
propagates into the translation. Use natural pauses and, if your microphone is noisy,
raise `--silence-ms` rather than cutting aggressively.

### Turning the numbers into perceived latency

For a 20-second utterance, processing takes ~1.8–2.0 seconds. For a short 3–5 second
utterance, ASR typically takes 0.3–0.6 seconds. Add the 600 ms VAD silence threshold and
the MT/TTS time; the typical “stop speaking → English audio” latency is roughly
**1–2 seconds**.

Model load after a warm cache, measured in a fresh process:

```text
ASR load   1.18 s   RSS 369 MB
TTS load   1.66 s   RSS 486 MB
TTS warmup 0.05 s
```

The first run additionally downloads ~464 MB (ASR), ~63 MB (voice), and ~77 MB (MT), and
performs a one-time CT2 conversion with torch.

## Caveats

- CPU frequency, core count, thermal throttling, and memory bandwidth all matter.
- ASR speed depends heavily on utterance length; Whisper processes 30-second windows.
- TTS speed depends on sentence length and whether ONNX Runtime has warmed up.
- RSS is process-level; shared libraries may be counted differently depending on how you
  measure. Use `danglish bench` and `scripts/speech_smoke.py` on your own machine.
