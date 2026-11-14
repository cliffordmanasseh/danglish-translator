# Holstebro Pinsekirke — Local Wi-Fi Audio Broadcast System
## Project Overview & "Where Is What" Guide (For External Review)

### 1. Purpose
An offline, zero-internet, low-latency live audio broadcast system running on a single Linux laptop (`Ubuntu / PipeWire / NetworkManager`). The laptop simultaneously acts as:
1. **Open Wi-Fi Access Point (`10.42.0.1/24`, no password)** via NetworkManager (`wlp193s0`). The network is open on purpose: anyone in church can join and listen.
2. **Audio Capture & Encoding Node** (`ffmpeg` capturing from PipeWire/PulseAudio or a wireless smartphone microphone and encoding to 48 kHz Opus + MP3 fallback).
3. **WebRTC & HTTP Broadcast Server** (`pion/webrtc/v4` in Go), fanning out a single encoded audio stream to 10–30+ smartphones over a single UDP port (`:8555`) with DSCP EF (`0xB8`) QoS marking.

---

### 2. File Map ("Where Is What")

All project files are located in:
`/data/cobalt8-twin-ferrum-archive/out/ferrum_g0/audio/`

| File / Directory | Role & Contents |
| :--- | :--- |
| **`start_church_broadcast.sh`** | **Turnkey Production Launcher.** Configures `dnsmasq` wildcard DNS (`address=/#/10.42.0.1`), opens UFW ports (`67/udp`, `53`, `80/tcp`, `8080/tcp`, `8443/tcp`, `8555/udp`; on firewalld only the last four), brings up the 2.4 GHz Ch 6 **open** Wi-Fi hotspot (`Church_Live_Audio`, no password), preserves the desktop user's PipeWire socket (`/run/user/1000/pulse/native`), runs the server, and restores normal Wi-Fi on `Ctrl+C`. Use `--local` to run on `127.0.0.1` without touching Wi-Fi. |
| **`main.go`** | **Backend Server (Go).** Embeds `index.html` and `speaker.html` into a single static binary. Supervises `ffmpeg`, ingests RTP on `127.0.0.1:5004`, normalizes RTP sequence numbers/timestamps, fans out to WebRTC listeners (`TrackLocalStaticRTP`), serves the HTTP MP3 fallback (`/stream.mp3`), responds to Apple/Android captive-portal probes, and generates terminal/SVG QR codes. |
| **`index.html`** | **Listener Web UI (Embedded in binary, served at `/` on ports `80` and `8080`).** Minimal church-themed UI (*Holstebro Pinsekirke*) with a single **Listen / Stop** button and live **dB** meter. Written in ES5 JS for compatibility with both modern and legacy iOS Safari / Android browsers. Connects via WebRTC (`/whep`) with a 4-second automatic fallback to `/stream.mp3` if WebRTC is unsupported. |
| **`speaker.html`** | **Speaker Web UI (Embedded in binary, served at `/speaker` on HTTP `:8080` and HTTPS `:8443`).** Minimal church-themed UI with a single **Speak / Stop** button and live **dB** meter. Unmutes the broadcast (`/api/mute`) and, when accessed over HTTPS (`:8443`), streams the smartphone's microphone via WebRTC (`/whip`) to all listeners. |
| **`church-audio-server`** | **Pre-compiled Static Binary (`CGO_ENABLED=0`).** Self-contained ~10 MB Linux executable with HTML assets embedded. Requires zero internet or compiler at runtime. |
| **`test_all_cases.sh`** | **Automated End-to-End Verification Script.** Starts an isolated local server instance and tests all 7 production paths: (1) Apple/Android Wi-Fi probes, (2) HTTP/HTTPS UI & catch-all routes, (3) WebRTC Listener `/whep`, (4) Speaker Phone `/whip` -> Listener `/whep` relay, (5) Legacy `/stream.mp3` validity via `ffprobe`, (6) Mute/Unmute API, and (7) mute keeps every stream alive (MP3 keeps flowing, WebRTC gets Opus silence frames, real audio returns on unmute). |
| **`QR_1_WiFi.pdf`**<br>**`QR_2_Listener.pdf`**<br>**`QR_3_Speaker.pdf`** | **Print-Ready A4 Vector PDFs.** Static QR code posters for (1) joining the open Wi-Fi hotspot (`WIFI:T:nopass;S:Church_Live_Audio;;`, no password), (2) opening the Listener player (`http://10.42.0.1:8080`), and (3) opening the Speaker console (`https://10.42.0.1:8443/speaker`). |
| **`go.mod` / `go.sum` / `vendor/`** | **Go Dependencies.** Fully vendored locally (`github.com/pion/webrtc/v4`, `github.com/pion/rtp`, `rsc.io/qr`) so offline builds (`go build -mod=vendor`) never touch the network. |

---

### 3. Key Code Sections in `main.go` (For Code Reviewers)

- **WebRTC Engine & Single-Port UDP Mux + DSCP QoS (`NewBroadcaster`, line 242):**
  - Binds UDP `:8555` (`SetICEUDPMux`), sets `SetNAT1To1IPs([]string{cfg.ServerIP}, webrtc.ICECandidateTypeHost)` with `ICEServers: []`, and marks socket packets with `IP_TOS = 0xB8` (DSCP Expedited Forwarding -> Wi-Fi WMM `AC_VO` voice queue).
- **Seamless RTP Normalization & Fan-Out (`writeAudioRTP`, line 327):**
  - Rewrites monotonically increasing `SequenceNumber` and continuous `Timestamp` across source switches (Laptop Mic `sourceID=1` vs. Speaker Phone `/whip` `sourceID=2`) before writing to the shared `webrtc.TrackLocalStaticRTP` and any active `/stream.mp3` listeners. While the laptop mic is muted it keeps sending packets but swaps the audio for a 3-byte Opus silence frame (`F8 FF FE`), so MP3 listeners never starve and WebRTC listeners stay fed.
- **Low-Jitter PipeWire/PulseAudio Capture (`superviseFFmpeg`, line 441):**
  - Runs `ffmpeg` with `-f pulse -fragment_size 1920` (eliminating 85 ms PulseAudio bursts down to steady 20 ms intervals), `-af asetpts=N/SR/TB` (sample-accurate monotonic PTS), and encodes 48 kHz mono Opus (`96 kbps`, `20 ms` frame duration) to `rtp://127.0.0.1:5004`.
- **OS Captive Portal & Connectivity Spoofing (`handleAppleHotspotProbe`, `handleAndroidProbe`, lines 534 and 541):**
  - Paired with `address=/#/10.42.0.1` in `dnsmasq` and Port 80 HTTP binding, replies `Success` to `captive.apple.com/hotspot-detect.html` and `204 No Content` to `connectivitycheck.gstatic.com/generate_204`. This prevents iOS and Android from dropping the offline Wi-Fi or routing Safari/Chrome over cellular data.
- **Universal Legacy Phone Fallback (`handleMP3Fallback`, line 561):**
  - On-demand live `libmp3lame` (`96 kbps`, `-flush_packets 1 -reservoir 0`) HTTP stream at `GET /stream.mp3`, fed from the live RTP fan-out so older phones without WebRTC still hear the broadcast. The listener page (`index.html`) has a watchdog: if the MP3 player stops advancing for 6 s, it reloads the stream by itself.

---

### 4. How a Reviewer Can Build & Test Locally

```bash
cd /data/cobalt8-twin-ferrum-archive/out/ferrum_g0/audio

# 1. Run the automated 7-test verification suite (does not modify Wi-Fi):
./test_all_cases.sh

# 2. Run the server locally on 127.0.0.1 (does not modify Wi-Fi):
./start_church_broadcast.sh --local

# 3. Run full production mode (starts Wi-Fi hotspot on 10.42.0.1):
./start_church_broadcast.sh
```
