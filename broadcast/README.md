# Church Wi-Fi broadcast

This directory contains a **read-only copy** of the Holstebro Pinsekirke /
Ferrum audio broadcast project, integrated into Danglish so the **English TTS**
can be heard on phones over Wi-Fi.

The original project at
`/data/cobalt8-twin-ferrum-archive/out/ferrum_g0/audio/` is untouched and still
contains the original binary, launcher, QR posters, and test suite. The copy in
`broadcast/church/` is what Danglish builds, starts, and supervises.

## Audio path

```text
Danish microphone
  → Whisper ASR
  → translation (OPUS-MT / EuroLLM)
  → tiny-voca English TTS
  → Pulse null sink "danglish_broadcast"
  → ffmpeg (Opus RTP)
  → Go broadcast server
  → WebRTC listeners + HTTP MP3 fallback
```

The app plays each English segment to **both** the operator output and the null
sink. The Go server captures `danglish_broadcast.monitor`, so the broadcast
contains the translation, not the Danish microphone.

## Usage

```bash
# Same Wi-Fi (LAN) mode: no sudo, no hotspot changes.
# The laptop's LAN IP is detected automatically.
danglish talk --broadcast

# Production mode: creates the open Church_Live_Audio hotspot (sudo),
# prints listener/QR URLs, and restores Wi-Fi on exit.
danglish talk --broadcast-hotspot

# Optional: explicit SSID / IP
danglish talk --broadcast --broadcast-ssid Church_Live_Audio --broadcast-ip 10.42.0.1
```

The listener page is served at `http://<laptop-ip>:8080/`; the operator QR
dashboard is `/qr`; the speaker page is `/speaker`.

`danglish broadcast` starts only the server (useful for testing the network or
for a second terminal):

```bash
danglish broadcast --ip 127.0.0.1 --http-port 18082
```

or in `danglish.toml`:

```toml
[broadcast]
enabled = true
local = true
server_ip = "auto"
ssid = "Church_Live_Audio"
http_port = 8080
https_port = 8443
webrtc_port = 8555
```

The desktop sidecar reads `DANGLISH_BROADCAST=1` (and optionally
`DANGLISH_BROADCAST_LOCAL=0` for hotspot mode), so the app can start the same
backend without a CLI flag.

## Requirements

- `pactl` (PipeWire/PulseAudio compatibility) and `ffmpeg`
- `sounddevice` / PortAudio (speech extras)
- `onnxruntime` + `dizione` for the tiny-voca English TTS
- Go only if you want to rebuild `church-audio-server`; the prebuilt binary is
  included

## Notes

- LAN mode depends on the Wi-Fi allowing client-to-client traffic. If the church
  Wi-Fi isolates clients, use `--broadcast-hotspot`, which creates its own open
  network.
- The hotspot is intentionally open; anyone in range can join and listen.
- The full-duplex speech loop can feed back through laptop speakers. Use
  `--half-duplex` or isolated audio hardware.
- The copied `QR_2_Listener.pdf` and `QR_3_Speaker.pdf` use the production
  hotspot address `10.42.0.1`. For LAN mode, open the live `/qr` dashboard so the
  QR code contains the detected LAN IP.
- The original `test_all_cases.sh` is retained in `broadcast/church/` for the
  standalone server verification suite.

## Layout

| Path | Purpose |
|---|---|
| `church/main.go` | Go WebRTC + MP3 server |
| `church/index.html` / `speaker.html` | Listener / speaker pages (embedded in the binary) |
| `church/start_church_broadcast.sh` | Production hotspot launcher |
| `church/church-audio-server` | Prebuilt Linux binary (also rebuilt by the launcher) |
| `church/vendor/` | Offline Go dependencies |
| `church/QR_*.pdf` | Printable posters |
| `../src/danglish/broadcast.py` | Python null-sink and server supervisor |
