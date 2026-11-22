# Danglish desktop

Chromium-based desktop shell for the local speech-to-speech pipeline. Electron renders
the UI; a bundled Python sidecar owns the models and audio devices.

```text
Electron main process
  ├─ renderer (HTML/CSS/JS)          captions, controls, settings
  └─ Python sidecar (PyInstaller)    ASR → EuroLLM → tiny-voca → audio
       └─ bundled models             resources/models
```

## Development

From the repository root:

```bash
uv sync --extra speech --extra convert --extra build
uv run danglish fetch

cd desktop
npm install
npm start
```

`main.js` looks for the repository `.venv` Python automatically:

- Linux/macOS: `../.venv/bin/python`
- Windows: `..\.venv\Scripts\python.exe`

Set `DANGLISH_PYTHON=/path/to/python` if you use a different environment. Set
`DANGLISH_DEVTOOLS=1` to open Chromium DevTools on launch.

The desktop app is fixed to the EuroLLM sermon translator and the church terminology
profile; there is no model or terminology selector. The app starts a local
`llama-server` in the background on first use and stops it when the app exits.

The default server and model paths are:

```text
/data/llama.cpp/build/bin/llama-server
/data/danglish_translator/models/eurollm-sermon-qwen-full-q5_k_m.gguf
```

Override them with `DANGLISH_LLAMA_SERVER` and `DANGLISH_LLAMA_MODEL`. The backend log
is written to `~/.config/danglish-desktop/backend.log`.

The UI starts the backend only when you click **Start listening**. Models load at that
point. The sidebar **Broadcast to phones** toggle serves the English TTS over the same
Wi-Fi and shows listener/QR links after startup. The **Mute** switch silences
local and broadcast audio without stopping captions.
Hotspot mode still uses the CLI: `danglish talk --broadcast-hotspot ...`.
The last 100 caption turns are kept in the app's local storage until you press
**Clear**; they never leave the machine. Closing the window does not lose the transcript.
The bundled ASR and tiny-voca models are reused automatically; EuroLLM runs from the
paths above.

If you already built the model bundle, the dev app reuses its ASR and tiny-voca models:

```text
desktop/resources/models/
```

Model files are platform-independent. To test on a Windows or macOS machine without
downloading anything, copy that directory from the Linux build into the same location on
the other machine. `npm start` and `npm run dist` both reuse it. The bundle is ~492 MB:
228 MB sermon MT, 238 MB Danish ASR, 26 MB tiny-voca.

## Build a distributable

```bash
cd desktop
npm run dist
```

That script:

1. builds the Python sidecar with PyInstaller into `resources/backend/`;
2. copies/downloads models into `resources/models/` (~492 MB);
3. runs electron-builder for the current platform into `release/`.

The app resources look like:

```text
resources/
  backend/danglish-backend/      Python runtime, ctranslate2, faster-whisper, onnxruntime
  models/sermon-opus-mt-int8/    sermon translation model
  models/whisper-small-danish/   Danish fine-tuned ASR model
  models/tiny-voca/              ONNX voice + dizione phonemizer
  legal/LICENSE
  legal/NOTICE
```

Installers are large (roughly 700 MB–1 GB depending on platform) because the Python
runtime and models are included. `resources/models` and `resources/backend` are generated
and are not committed.

## Windows and low-memory machines

On Windows, or on any machine with less than 6 GB RAM, the desktop app automatically
uses the bundled **OPUS-MT sermon edge model** instead of EuroLLM. That model is
228 MB int8, ~304 MB RSS, and does not need `llama-server`.

The Windows build is produced by the **Desktop builds** workflow. Broadcast is
Linux-only in this version and is hidden on Windows; translation, captions, and
tiny-voca speech work normally.

## Cross-platform builds

Electron can build for the current OS; PyInstaller cannot cross-compile a Python sidecar.
Use the `Desktop builds` GitHub Actions workflow to produce:

- Linux x64 AppImage
- Windows x64 NSIS
- macOS arm64 and x64 DMG/zip

Builds are unsigned by default. macOS may require right-click → Open or
`xattr -dr com.apple.quarantine Danglish.app`. Windows SmartScreen may warn on an unsigned
installer.

## Microphone permissions

- **macOS:** the app declares `NSMicrophoneUsageDescription` and the audio-input
  entitlement. The first start triggers the system permission prompt. If it is denied,
  enable Danglish under System Settings → Privacy & Security → Microphone.
- **Windows:** allow microphone access under Settings → Privacy → Microphone.
- **Linux:** PipeWire/PulseAudio access is per-user. The sidecar bundles `libportaudio`
  when available on the build machine; if your distribution blocks it, install
  `libportaudio2`.

Audio is captured by the Python sidecar, not by the Chromium renderer, so there is no
browser `getUserMedia` prompt.

## Audio interfaces

The UI exposes input/output device names plus input/output channel numbers under
**Voice settings → Advanced audio settings**. Prefer device names over indices —
PortAudio indices can change when PipeWire nodes are reordered. Click **Refresh devices**
after plugging in an interface.

For a UAD Volt 4, input 1 is normally channel 0 and input 2 is channel 1. Full-duplex is
always on in the desktop app; use headphones or isolated audio I/O. Alternatively, run
`danglish audio-check` in a terminal to see per-channel RMS levels and play a test tone.

## Troubleshooting

- **“Python environment not found”** — run `uv sync` at the repository root, or set
  `DANGLISH_PYTHON`.
- **EuroLLM server/model not found** — set `DANGLISH_LLAMA_SERVER` and
  `DANGLISH_LLAMA_MODEL`, or place them at the default paths above.
- **Backend stopped** — read `~/.config/danglish-desktop/backend.log` for the last run.
- **No audio from the UI** — check the device names/channels, the interface monitor knob,
  and the OS microphone permission. Test with `danglish audio-check`.
- **Electron sandbox error on Linux** — run the AppImage with `--no-sandbox` for local
  testing, or install the distribution's sandbox support.
- **Install size** — models are bundled intentionally. A first-run download variant would
  reduce the installer but would not be fully offline.

## Memory

The UI process is Chromium, so it adds memory over the CLI. On the reference Linux machine
the loaded Python speech stack is ~0.6 GB RSS and the Electron processes sum to ~0.6 GB
(summed RSS overcounts shared pages). A loaded window is therefore ~1.1–1.2 GB, and a
peak ASR turn can approach ~1.5 GB total. That still fits under 2 GB, but with limited
headroom for the OS. The CLI (`danglish talk`) remains the low-memory option.

## Licensing

The desktop code is Apache-2.0. The bundled tiny-voca model, dizione, and ONNX Runtime
are Apache-2.0/MIT; the optional Piper fallback is not bundled. The EuroLLM fine-tuned
weights are research-use-only. The app ships `LICENSE` and `NOTICE` under
`resources/legal` and exposes them from the Licenses button. If you distribute a built
app, retain the model and phonemizer attributions in `NOTICE`.
