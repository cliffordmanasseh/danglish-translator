# Contributing

Thanks for helping build a local Danish → English interpreter. Please keep the project
small, inspectable, and friendly to a 2 GB machine.

## Setup

```bash
git clone <repo> danglish
cd danglish
uv sync --all-extras
uv run --no-sync danglish fetch
```

The `fetch` step downloads the OPUS-MT model and converts it with torch. Commit only
code and docs; `models/` and voice files are ignored.

For a lean text-only setup:

```bash
uv sync --extra dev
uv run --extra convert danglish fetch
```

For speech development:

```bash
uv sync --extra dev --extra server --extra speech --extra convert
```

On Debian/Ubuntu, microphone playback needs PortAudio:

```bash
sudo apt install libportaudio2
```

## Tests and checks

```bash
uv run pytest -m "not integration"     # fast, no model or audio devices
uv run pytest                          # includes model/voice integration tests
uv run ruff check .
uv run ruff format --check .
```

Integration tests are marked `integration` and skip gracefully when a model or optional
package is missing. The TTS integration test downloads the selected Piper voice on first
run.

To test the audio pipeline without a microphone:

```bash
uv run --no-sync python scripts/speech_smoke.py your-danish-clip.wav -o english.wav
```

Use a mono 16-bit PCM WAV. The script uses the same ASR → MT → TTS path as `danglish talk`.
Please do not commit copyrighted audio; use a short clip you have the right to use, or
synthesize one.

## Desktop app

The Electron UI lives in `desktop/`. It is a thin shell around the Python CLI:

```bash
cd desktop
npm install
npm start                 # uses ../.venv/bin/python (or ..\.venv\Scripts\python.exe)
```

Set `DANGLISH_PYTHON` to use a different interpreter. The UI starts the backend only when
the user clicks Start; the backend writes JSON events to stdout and logs to stderr.

To build an installer:

```bash
cd ..
uv sync --extra speech --extra convert --extra build
cd desktop
npm run dist
```

This builds the PyInstaller sidecar, bundles models into `resources/models/`, and runs
electron-builder. Generated models, the sidecar, `node_modules`, and `release/` are ignored.
PyInstaller and electron-builder must run on the target OS; use the `Desktop builds`
workflow for Windows/macOS artifacts. Unsigned macOS builds need right-click → Open during
testing.

When changing the backend interface, check `desktop/main.js`'s argument construction and
the JSON shape consumed by `desktop/renderer/app.js`. When adding a model, update
`scripts/bundle_models.py`, `NOTICE`, and the desktop resource paths.

## Performance and memory rules

- The hard target is **2 GB peak RSS for the full speech stack**; the current measured
  peak is ~1.0 GB. The Electron shell adds ~0.6 GB across Chromium processes, so a
  desktop peak approaches ~1.5 GB. New features should leave meaningful headroom.
- Do not add torch, transformers, or other heavy packages to the core dependencies.
  They belong in optional extras.
- Report `danglish bench` output and, for audio changes, the RSS/peak and ASR/MT/TTS
  timings printed by `speech_smoke.py` in the pull request.
- Cache or reuse models within a process; do not load a second copy for a new request.
- Prefer CPU-friendly integer inference paths (CTranslate2 int8, ONNX Runtime).

## Adding a model or voice

- Translation models must be convertible to CTranslate2 and use the Marian +
  SentencePiece layout expected by `backends/ct2.py` (or you must add an adapter).
- Remember the source `</s>` token for Marian models; see `docs/models.md`.
- Update `NOTICE` and `docs/models.md` with license and attribution.
- Do not commit model weights or voice files.

## Code style

- Python 3.10+; `from __future__ import annotations` is used throughout.
- Ruff for linting and formatting (`line-length = 100`).
- Keep public functions typed and errors user-facing via `DanglishError` subclasses.
- Optional dependencies must be imported lazily and raise `DependencyMissingError` with
  an install hint.
- Prefer small, testable units. `SpeechTranslator` is deliberately device-free so the
  audio path can be tested with fakes.

## Pull request checklist

- [ ] Tests added or updated
- [ ] `ruff check` and `ruff format --check` pass
- [ ] No model/audio artifacts committed
- [ ] Memory and latency impact stated for performance-sensitive changes
- [ ] Licenses and NOTICE updated if a model, voice, or dependency was added
