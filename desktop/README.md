# Danglish desktop

Chromium-based desktop shell for the local speech-to-speech pipeline. Electron renders
the UI; a bundled Python sidecar owns the models and audio devices.

```text
Electron main process
  ├─ renderer (HTML/CSS/JS)          captions, controls, settings
  └─ Python sidecar (PyInstaller)    ASR → OPUS-MT → Piper → audio
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

The UI starts the backend only when you click **Start listening**. Models load at that
point. The last 100 caption turns are kept in the app's local storage until you press
**Clear**; they never leave the machine. Closing the window does not lose the transcript. In development, the Danish Whisper fine-tune and the Piper voice are downloaded on
first use; the MT model must already exist from `danglish fetch`.

If you already built the model bundle (~375 MB), the dev app uses it automatically:

```text
desktop/resources/models/
```

Model files are platform-independent. To test on a Windows or macOS machine without
downloading anything, copy that directory from the Linux build into the same location on
the other machine. `npm start` and `npm run dist` both reuse it. The bundle is ~375 MB:
77 MB MT, 238 MB Danish ASR, 60 MB voice.

## Build a distributable

```bash
cd desktop
npm run dist
```

That script:

1. builds the Python sidecar with PyInstaller into `resources/backend/`;
2. copies/downloads models into `resources/models/` (~375 MB);
3. runs electron-builder for the current platform into `release/`.

The app resources look like:

```text
resources/
  backend/danglish-backend/      Python runtime, ctranslate2, faster-whisper, Piper, onnxruntime
  models/opus-mt-da-en-int8/     translation model
  models/whisper-small-danish/   Danish fine-tuned ASR model
  models/voices/                 Piper voice
  legal/LICENSE
  legal/NOTICE
```

Installers are large (roughly 700 MB–1 GB depending on platform) because the Python
runtime and models are included. `resources/models` and `resources/backend` are generated
and are not committed.

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

The UI exposes input/output device names plus input/output channel numbers. Prefer device
names over indices — PortAudio indices can change when PipeWire nodes are reordered. In
the app, click **Audio & voice settings → Refresh devices** after plugging in an interface.

For a UAD Volt 4, input 1 is normally channel 0 and input 2 is channel 1. Full-duplex is
the default; turn it off in the sidebar for laptop speakers. Alternatively, run
`danglish audio-check` in a terminal to see per-channel RMS levels and play a test tone.

## Troubleshooting

- **“Python environment not found”** — run `uv sync` at the repository root, or set
  `DANGLISH_PYTHON`.
- **Model not found** — run `uv run danglish fetch` in the repository root.
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

The desktop code is Apache-2.0. The bundled `piper-tts` runtime is GPL-3.0-or-later, and
the default English voice is CC BY 4.0. The app ships `LICENSE` and `NOTICE` under
`resources/legal` and exposes them from the Licenses button. If you distribute a built app,
comply with the GPL and voice attribution requirements.
