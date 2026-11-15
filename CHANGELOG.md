# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses semantic versioning.

## [Unreleased]

### Added

- Wi-Fi broadcast: `danglish talk --broadcast` / `--broadcast-hotspot` routes
  English TTS to a Pulse null sink and serves it over the copied church
  WebRTC/MP3 server; printable QR posters and a dynamic `/qr` dashboard.
- Fine-tuned OPUS-MT tc-big sermon edge model (int8): 228 MB, 304 MB RSS,
  143 ms p50, 0.8658 COMET; recommended edge default.
- `release/sermon-opus-model/` model card, checksums, and archive.
- Optional `llama` backend for the EuroLLM-1.7B sermon translation model
  (Q5_K_M): 0.8663 COMET on 60 held-out sermons, ~1.29 GB RSS, ~270 ms p50.
- `release/sermon-model/` with model card, checksums, and a dependency-free
  client; `DANGLISH_BACKEND=llama` selects the backend.

### Changed

- Replaced the default Piper English voice with tiny-voca M2: 21 MB ONNX +
  5 MB Apache-2.0 phonemizer, ~70–85× real time, and ~169 MB RSS instead of
  ~248 MB. Piper remains available as the optional `tts-piper` fallback
  (`--tts-backend piper`).
- Low-latency chunking: cut after 300 ms of silence or 4 s of continuous speech (was
  600 ms / 15 s), so English audio starts within roughly two seconds of each chunk instead
  of waiting for a long monologue to end.
- Full-duplex is now the default; `--half-duplex` (or the desktop toggle) mutes the
  microphone during playback for laptop speakers.
- The desktop app shows a “Translating N s of speech…” status as soon as a chunk closes,
  and the context selector is labelled “Terminology”.

## [0.1.0] — 2026-11-11

### Added

- Streaming Danish → English text translation with sentence-aware segmentation.
- CTranslate2 int8 backend for `Helsinki-NLP/opus-mt-da-en`, plus model download and
  conversion (`danglish fetch`).
- Local WebSocket server and text web UI with partial translations.
- Speech-to-speech path: energy VAD, faster-whisper Danish ASR, OPUS-MT, Piper English
  TTS, and microphone playback (`danglish talk`).
- Default ASR is the MIT-licensed `lucasw89/whisper-small-danish-ct2-int8` Danish
  fine-tune (pre-quantized int8, ~238 MB vs ~464 MB stock small), with a Silero
  speech-presence gate to reject silence/noise hallucinations without trimming the
  utterance. This substantially improves the Danish transcript that feeds translation.
- Speech translation now translates the whole ASR turn by default, preserving context
  across sentences; `--sentence-level` restores per-sentence translation. This fixed
  context-dependent errors such as “indtalt” → “entered” vs “recorded”.
- Parallel audio runtime with separate capture/VAD, translation, and playback threads and
  bounded queues. `--full-duplex` keeps listening while English audio plays for isolated
  input/output setups; half-duplex remains the default for laptop speakers.
- `danglish audio-check`: lists devices, records a few seconds, reports per-channel input
  levels, and plays an output test tone.
- Context profiles (`general`, `sermon`) with conservative terminology preferences and a
  `--context` option in the CLI.
- Chromium/Electron desktop app with a bundled Python sidecar and all three models, a
  caption UI, device/channel settings, context selection, full-duplex control, and a
  Linux AppImage build. Cross-platform packaging workflow for Windows and macOS.
- Stable audio-device selection by name (`--input-device "Volt 4"`), native-rate capture
  with resampling to 16 kHz, explicit output streams for multi-channel interfaces, and
  `--input-channel` / `--output-channel` mapping. This fixes UAD/PipeWire setups where
  PortAudio's `sd.play()` hung and where forcing 16 kHz bypassed the interface.
- CLI commands: `translate`, `repl`, `stream`, `serve`, `talk`/`mic`, `fetch`, `doctor`,
  `bench`, `version`.
- Memory diagnostics and a benchmark that fails when peak RSS exceeds the 2 GB budget.
- Optional extras for server, ASR, TTS, speech, conversion, and development.
- Unit and integration tests, including a regression test for the Marian source-EOS
  repetition bug.
- Architecture, model, benchmark, contribution, and license documentation.

### Verified

- Text pipeline: 147 MB peak RSS, p50 36 ms per sentence.
- Speech pipeline: 891 MB peak RSS for one 20 s turn (999 MB in a five-turn session),
  20 s Danish clip → English audio in ~2.0 s (ASR 1.46 s, MT 0.15 s, TTS 0.38 s).
