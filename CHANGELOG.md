# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses semantic versioning.

## [0.1.0] — 2026-11-11

### Added

- Streaming Danish → English text translation with sentence-aware segmentation.
- CTranslate2 int8 backend for `Helsinki-NLP/opus-mt-da-en`, plus model download and
  conversion (`danglish fetch`).
- Local WebSocket server and text web UI with partial translations.
- Speech-to-speech path: energy VAD, faster-whisper Danish ASR, OPUS-MT, Piper English
  TTS, and microphone playback (`danglish talk`).
- Parallel audio runtime with separate capture/VAD, translation, and playback threads and
  bounded queues. `--full-duplex` keeps listening while English audio plays for isolated
  input/output setups; half-duplex remains the default for laptop speakers.
- `danglish audio-check`: lists devices, records a few seconds, reports per-channel input
  levels, and plays an output test tone.
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
- Speech pipeline: 876 MB peak RSS for one 20 s turn (1,024 MB in a longer multi-turn
  session), 20 s Danish clip → English audio in ~1.8 s (ASR 1.3 s, MT 0.12 s,
  TTS 0.35 s).
