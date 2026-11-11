# Architecture

## Goals and constraints

Danglish is a local, open-source Danish → English translator for small, CPU-only
machines. The primary interface is **speech in, speech out**; a text path exists for
debugging, scripting, and the web demo.

Hard constraints:

1. **The full speech-to-speech process should fit comfortably below 2 GB of RSS.**
   Optional engines are loaded lazily; the text-only path is intentionally tiny.
2. **No network access at inference time.** Models are downloaded once, then used from
   local disk.
3. **CPU only.** CTranslate2 int8 for translation and ASR, ONNX Runtime for TTS.
4. **Boring, inspectable pieces.** Each stage has a narrow interface and can be tested
   without a microphone.

Non-goals for this version: simultaneous interpreting, multi-speaker diarization,
document-level context, and cloud-scale throughput.

## Speech-to-speech data flow

```text
┌──────────────┐
│ microphone   │  16 kHz mono float32, 30 ms frames
└──────┬───────┘
       │
       ▼
┌──────────────┐  speech starts after 3 consecutive speech frames
│ EnergyVAD    │  utterance ends after 600 ms of silence (or 15 s max)
│ + pre-roll   │  keeps 500 ms of pre-roll and 150 ms of tail
└──────┬───────┘
       │  utterance audio (~1–15 s)
       ▼
┌──────────────┐  faster-whisper "small", int8, language="da"
│ Whisper ASR  │  no internal VAD, no previous-text conditioning
└──────┬───────┘
       │  Danish transcript
       ▼
┌──────────────┐  sentence segmentation + CTranslate2 opus-mt-da-en int8
│ MT pipeline  │  batch finals, source EOS appended
└──────┬───────┘
       │  one or more English sentences
       ▼
┌──────────────┐  Piper voice, 22.05 kHz mono
│ Piper TTS    │  one synthesis per sentence
└──────┬───────┘
       │  float32 waveform
       ▼
┌──────────────┐
│ speaker      │  microphone is muted while audio plays
└──────────────┘
```

### Why two-stage (ASR → MT) instead of Whisper's built-in translation?

Whisper can translate speech directly to English, but the two-stage design is better here:

- the Danish transcript is visible, so users can see what was recognized;
- the translation model can be swapped or fine-tuned independently;
- sentence-level TTS can start per translated sentence;
- the same MT core serves the text path and the web demo.

The cost is one extra model and a small amount of latency. The measured MT cost is
~0.1 s for a 20-second utterance, so it is not the bottleneck.

## Streaming text path

The text pipeline is a small state machine:

1. `SentenceSegmenter.feed(delta)` appends incoming characters and emits complete
   sentences. It is conservative: it waits when a boundary is ambiguous rather than
   making a wrong cut. It understands common Danish abbreviations (`bl.a.`, `f.eks.`,
   `kl.`, `osv.`), thousands separators (`1.000`), times (`14.30`), initials
   (`J. K. Rowling`), ellipses, and quoted speech.
2. Complete sentences are translated in a batch. IDs increase monotonically.
3. The unfinished tail can be translated as a **partial** preview after a debounce.
   A new delta invalidates the preview.
4. `flush()` commits the remaining tail. This is what happens at the end of an ASR
   utterance or when the user presses the “Translate now” button.
5. If a stream never produces punctuation, the pending buffer is force-split at a
   clause boundary after `max_pending_chars` (default 4000).

The WebSocket demo uses the same pipeline per client session. A version counter prevents
stale partials from being sent after new input or a reset.

### Segmentation trade-offs

The segmenter deliberately accepts some false negatives:

- an abbreviation that genuinely ends a sentence may be held until flush;
- a period followed by a lowercase word is treated as a continuation;
- headings and list items without terminal punctuation are only emitted on flush.

These are preferable to cutting a sentence in the middle, which produces visibly wrong
translations.

## Model details and the source-EOS pitfall

The OPUS-MT model is a Marian encoder-decoder. Hugging Face's `MarianTokenizer` appends
`</s>` to the **source** token sequence. CTranslate2 does not; its converted config sets
`add_source_eos: false`. If the token is omitted, the model degenerates into repetition
loops (“Good morning morning morning …”) until it hits the decoding limit.

Danglish appends `</s>` explicitly in `backends/ct2.py`. This is covered by an integration
regression test. Keep it in mind when adding another Marian model.

Translation inputs longer than 500 source tokens are split recursively at word
boundaries as a safety net. Normal streaming input is already sentence-sized.

## Concurrency and the half-duplex rule

The `talk` path is single-threaded:

- PortAudio calls the capture callback on its own thread; frames go into a bounded queue.
- The main loop runs VAD, ASR, MT, and TTS sequentially.
- While English audio is playing, a `muted` event makes the capture callback drop frames.
  The queue is drained and the VAD is reset after playback. This prevents the microphone
  from transcribing the translator's own voice.
- The user cannot speak over the translation. This is intentional for a first version;
  echo cancellation and full-duplex barge-in are roadmap items.

The web server is asynchronous around synchronous CPU work: translation calls run in
`asyncio.to_thread`, and each WebSocket session has a lock so its pipeline mutations are
serialized. Multiple sessions share the backend; CTranslate2 is safe to call from
multiple threads, but a local single-user setup is the target.

## Memory model

Measured on the development machine:

- text pipeline: **147 MB** peak;
- text + FastAPI server: **185 MB** peak;
- speech stack (ASR small + MT + Piper): **~1.0 GB** peak, stabilizing around 925 MB RSS
  after several turns.

The speech peak includes CTranslate2's temporary allocation while quantizing the Whisper
weights on load (~730 MB), growing to ~1,024 MB as ASR/TTS buffers are exercised. The
budget check uses peak RSS, which is the number that matters on a 2 GB machine. The
measured peak leaves around 1 GB of headroom for the OS.

Memory reduction levers:

- use a smaller ASR model (`base`, `tiny`) at a significant Danish quality cost;
- use greedy translation (`beam_size = 1`);
- cap threads (2 threads is enough on a small PC);
- use the text-only path when audio is not needed.

## Failure handling

- Missing model → a clear `danglish fetch` hint, exit code 1.
- Missing optional dependency → `uv sync --extra speech` / `pip install "danglish[speech]"`.
- TTS failure → the English text is still printed; the turn continues.
- Empty ASR result → the turn is skipped.
- Audio device failure → a clean error with a `--list-devices` hint.
- A backend that returns the wrong number of translations raises `BackendError` rather
  than silently misaligning events.

## Security

The web server binds to `127.0.0.1` by default and has no authentication. Do not expose
it to a network without putting an authenticated proxy in front of it. Audio is processed
in memory and is not written to disk by `talk`; model downloads go through the Hugging
Face cache. There is no telemetry.

## Testing strategy

- Unit tests use fake backends, fake ASR/TTS engines, and synthetic frames; no models
  or audio devices are required.
- Segmentation tests cover Danish abbreviations, numbers, quotes, and incremental input.
- Integration tests are marked `integration` and require a converted MT model; the TTS
  integration test also needs Piper and a downloaded voice.
- `scripts/speech_smoke.py` exercises the same `SpeechTranslator` used by `talk` on a WAV
  file, which is useful for testing a new machine without a microphone.
