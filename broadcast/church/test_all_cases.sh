#!/usr/bin/env bash
# ==============================================================================
# Automated End-to-End Verification Suite for Holstebro Pinsekirke Audio System
# ==============================================================================
# Tests all 7 production scenarios locally on this PC without touching Wi-Fi:
#   1. Apple iOS & Android Captive Portal / Connectivity Probes
#   2. Listener, Speaker, QR, and Catch-All HTTP/HTTPS Pages
#   3. Live WebRTC Listener Stream (/whep receiving live Opus RTP from PC mic)
#   4. Wireless Speaker Phone Mic (/whip -> Server -> Listener /whep relay)
#   5. Legacy iPhone/Android HTTP MP3 Fallback (/stream.mp3 verified via ffprobe)
#   6. Speaker Mute / Unmute Control (/api/mute)
#   7. Mute keeps every stream alive: MP3 keeps flowing, WebRTC gets Opus silence,
#      real audio returns on unmute (negative control: fails on the pre-fix code)
# ==============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PATH="/home/clmm/.local/go/bin:/usr/local/go/bin:$PATH"

echo "======================================================================"
echo "  Building ./church-audio-server (offline vendor build)..."
echo "======================================================================"
CGO_ENABLED=0 go build -mod=vendor -ldflags="-s -w" -o church-audio-server main.go

TEST_HTTP=18080
TEST_HTTPS=18443
TEST_WEBRTC=18555

./church-audio-server \
  --ip=127.0.0.1 \
  --http-port=${TEST_HTTP} \
  --https-port=${TEST_HTTPS} \
  --webrtc-port=${TEST_WEBRTC} \
  --rtp-port=15004 \
  --print-qr=false >/tmp/church_test_suite.log 2>&1 &
SRV_PID=$!

cleanup() {
  kill "${SRV_PID}" 2>/dev/null || true
  wait "${SRV_PID}" 2>/dev/null || true
  rm -f /tmp/church_test_suite.log /tmp/test_fallback.mp3 /tmp/e2e_webrtc_verify.go /tmp/test_fallback_muted.mp3 /tmp/e2e_mute_verify.go
}
trap cleanup EXIT INT TERM

sleep 1.5

echo ""
echo "[TEST 1/7] Apple iOS & Android Wi-Fi Connectivity Probes..."
APPLE_OUT=$(curl -fsS "http://127.0.0.1:${TEST_HTTP}/hotspot-detect.html")
if [[ "${APPLE_OUT}" == *"Success"* ]]; then
  echo "  -> PASS: Apple iOS probe (/hotspot-detect.html) returned 'Success'"
else
  echo "  -> FAIL: Unexpected Apple probe output: ${APPLE_OUT}"
  exit 1
fi

ANDROID_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${TEST_HTTP}/generate_204")
if [[ "${ANDROID_CODE}" == "204" ]]; then
  echo "  -> PASS: Android probe (/generate_204) returned HTTP 204"
else
  echo "  -> FAIL: Unexpected Android probe HTTP code: ${ANDROID_CODE}"
  exit 1
fi

echo ""
echo "[TEST 2/7] Web UI Pages (HTTP, HTTPS, QR, and Catch-All)..."
for endpoint in "/" "/speaker" "/qr" "/any-typed-path"; do
  TITLE=$(curl -fsS "http://127.0.0.1:${TEST_HTTP}${endpoint}" | grep -o "<title>Holstebro Pinsekirke</title>" || true)
  if [[ -n "${TITLE}" ]]; then
    echo "  -> PASS: GET ${endpoint} served 'Holstebro Pinsekirke'"
  else
    echo "  -> FAIL: GET ${endpoint} did not return expected page"
    exit 1
  fi
done

HTTPS_TITLE=$(curl -k -fsS "https://127.0.0.1:${TEST_HTTPS}/speaker" | grep -o "<title>Holstebro Pinsekirke</title>" || true)
if [[ -n "${HTTPS_TITLE}" ]]; then
  echo "  -> PASS: HTTPS GET /speaker (:8443) served 'Holstebro Pinsekirke'"
else
  echo "  -> FAIL: HTTPS GET /speaker failed"
  exit 1
fi

echo ""
echo "[TEST 3/7 & 4/7] Live WebRTC Listener (/whep) + Speaker Phone Mic (/whip) Relay..."
cat << 'EOF' > /tmp/e2e_webrtc_verify.go
//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
)

func main() {
	baseURL := "http://127.0.0.1:18080"

	// Connect Listener via /whep and verify live PC mic Opus RTP arrives
	listenerPC, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		panic(err)
	}
	defer listenerPC.Close()

	_, _ = listenerPC.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio, webrtc.RTPTransceiverInit{
		Direction: webrtc.RTPTransceiverDirectionRecvonly,
	})

	gotMicRTP := make(chan int, 1)
	gotPhoneMarker := make(chan bool, 1)

	listenerPC.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
		for {
			pkt, _, err := track.ReadRTP()
			if err != nil {
				return
			}
			select {
			case gotMicRTP <- len(pkt.Payload):
			default:
			}
			if bytes.Equal(pkt.Payload, []byte{0xCA, 0xFE, 0xBA, 0xBE, 0x11, 0x22, 0x33, 0x44}) {
				select {
				case gotPhoneMarker <- true:
				default:
				}
				return
			}
		}
	})

	lOffer, _ := listenerPC.CreateOffer(nil)
	lGather := webrtc.GatheringCompletePromise(listenerPC)
	_ = listenerPC.SetLocalDescription(lOffer)
	<-lGather

	lBody, _ := json.Marshal(map[string]string{"type": "offer", "sdp": listenerPC.LocalDescription().SDP})
	lResp, err := http.Post(baseURL+"/whep", "application/json", bytes.NewReader(lBody))
	if err != nil {
		panic(err)
	}
	var lAns struct {
		Type string `json:"type"`
		SDP  string `json:"sdp"`
	}
	_ = json.NewDecoder(lResp.Body).Decode(&lAns)
	lResp.Body.Close()
	_ = listenerPC.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: lAns.SDP})

	select {
	case n := <-gotMicRTP:
		fmt.Printf("  -> PASS [TEST 3/7]: Listener received live WebRTC Opus RTP packet from PC mic (%d bytes payload)\n", n)
	case <-time.After(4 * time.Second):
		panic("FAIL: Timed out waiting for PC mic WebRTC RTP packet")
	}

	// Now connect simulated Speaker Phone via /whip and verify relay to Listener
	speakerPC, _ := webrtc.NewPeerConnection(webrtc.Configuration{})
	defer speakerPC.Close()

	speakerTrack, _ := webrtc.NewTrackLocalStaticRTP(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2},
		"speaker-mic", "speaker-stream",
	)
	_, _ = speakerPC.AddTrack(speakerTrack)

	sOffer, _ := speakerPC.CreateOffer(nil)
	sGather := webrtc.GatheringCompletePromise(speakerPC)
	_ = speakerPC.SetLocalDescription(sOffer)
	<-sGather

	sBody, _ := json.Marshal(map[string]string{"type": "offer", "sdp": speakerPC.LocalDescription().SDP})
	sResp, err := http.Post(baseURL+"/whip", "application/json", bytes.NewReader(sBody))
	if err != nil {
		panic(err)
	}
	var sAns struct {
		Type string `json:"type"`
		SDP  string `json:"sdp"`
	}
	_ = json.NewDecoder(sResp.Body).Decode(&sAns)
	sResp.Body.Close()
	_ = speakerPC.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: sAns.SDP})

	go func() {
		var seq uint16
		var ts uint32
		for i := 0; i < 150; i++ {
			seq++
			ts += 480
			_ = speakerTrack.WriteRTP(&rtp.Packet{
				Header: rtp.Header{
					Version:        2,
					PayloadType:    111,
					SequenceNumber: seq,
					Timestamp:      ts,
					SSRC:           99999,
				},
				Payload: []byte{0xCA, 0xFE, 0xBA, 0xBE, 0x11, 0x22, 0x33, 0x44},
			})
			time.Sleep(15 * time.Millisecond)
		}
	}()

	select {
	case <-gotPhoneMarker:
		fmt.Println("  -> PASS [TEST 4/7]: Wireless Speaker Phone (/whip) -> Server -> Listener (/whep) relay verified")
	case <-time.After(4 * time.Second):
		panic("FAIL: Timed out waiting for Speaker Phone relay packet")
	}
}
EOF
go run -mod=vendor /tmp/e2e_webrtc_verify.go

echo ""
echo "[TEST 5/7] Legacy iPhone/Android HTTP MP3 Fallback (/stream.mp3)..."
timeout 2s curl -s "http://127.0.0.1:${TEST_HTTP}/stream.mp3" > /tmp/test_fallback.mp3 || true
MP3_INFO=$(ffprobe -hide_banner /tmp/test_fallback.mp3 2>&1 | grep "Audio: mp3" || true)
if [[ -n "${MP3_INFO}" ]]; then
  echo "  -> PASS: /stream.mp3 decoded as valid live MP3 stream: ${MP3_INFO}"
else
  echo "  -> FAIL: /stream.mp3 did not produce valid MP3 audio"
  exit 1
fi

echo ""
echo "[TEST 6/7] Speaker Mute / Unmute Control (/api/mute)..."
curl -fsS -X POST -H "Content-Type: application/json" -d '{"muted":true}' "http://127.0.0.1:${TEST_HTTP}/api/mute" >/dev/null
MUTED_STATE=$(curl -fsS "http://127.0.0.1:${TEST_HTTP}/stats" | grep -o '"muted":true' || true)
curl -fsS -X POST -H "Content-Type: application/json" -d '{"muted":false}' "http://127.0.0.1:${TEST_HTTP}/api/mute" >/dev/null
UNMUTED_STATE=$(curl -fsS "http://127.0.0.1:${TEST_HTTP}/stats" | grep -o '"muted":false' || true)
if [[ -n "${MUTED_STATE}" && -n "${UNMUTED_STATE}" ]]; then
  echo "  -> PASS: Mute & Unmute API verified"
else
  echo "  -> FAIL: Mute state toggle failed"
  exit 1
fi


echo ""
echo "[TEST 7/7] Mute keeps every listener's stream alive (silence, not a dead stream)..."
# Why: the Speaker page's Stop button mutes the laptop mic. If mute stopped sending
# packets, MP3-fallback phones would stall for good and WebRTC phones would get
# nothing. Mute must send Opus silence frames instead.
curl -fsS -X POST -H "Content-Type: application/json" -d '{"muted":true}' "http://127.0.0.1:${TEST_HTTP}/api/mute" >/dev/null

# 7a. MP3 listener while muted. Limit: /stream.mp3 is 96 kbit/s = 12,000 bytes/s.
# We capture for 4 s and require at least 1 s of audio (12,000 bytes), which leaves
# 3 s for ffmpeg start-up. A dead stream gives 0 bytes.
timeout 4s curl -s "http://127.0.0.1:${TEST_HTTP}/stream.mp3" > /tmp/test_fallback_muted.mp3 || true
MUTED_MP3_BYTES=$(stat -c %s /tmp/test_fallback_muted.mp3 2>/dev/null || echo 0)
if [[ "${MUTED_MP3_BYTES}" -ge 12000 ]]; then
  echo "  -> PASS [7a]: /stream.mp3 kept flowing while muted (${MUTED_MP3_BYTES} bytes in 4 s, need >= 12000)"
else
  echo "  -> FAIL [7a]: /stream.mp3 starved while muted (${MUTED_MP3_BYTES} bytes in 4 s, need >= 12000)"
  exit 1
fi

# 7b/7c. WebRTC listener: silence frames while muted, real audio after unmute.
cat << 'EOF' > /tmp/e2e_mute_verify.go
//go:build ignore

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/pion/webrtc/v4"
)

func setMute(m bool) {
	b, _ := json.Marshal(map[string]bool{"muted": m})
	r, err := http.Post("http://127.0.0.1:18080/api/mute", "application/json", bytes.NewReader(b))
	if err != nil {
		panic(err)
	}
	r.Body.Close()
}

func main() {
	silence := []byte{0xF8, 0xFF, 0xFE}
	pc, err := webrtc.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		panic(err)
	}
	defer pc.Close()
	_, _ = pc.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio, webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly})

	payloads := make(chan []byte, 512)
	pc.OnTrack(func(t *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		for {
			p, _, err := t.ReadRTP()
			if err != nil {
				return
			}
			select {
			case payloads <- append([]byte(nil), p.Payload...):
			default:
			}
		}
	})
	offer, _ := pc.CreateOffer(nil)
	g := webrtc.GatheringCompletePromise(pc)
	_ = pc.SetLocalDescription(offer)
	<-g
	body, _ := json.Marshal(map[string]string{"type": "offer", "sdp": pc.LocalDescription().SDP})
	resp, err := http.Post("http://127.0.0.1:18080/whep", "application/json", bytes.NewReader(body))
	if err != nil {
		panic(err)
	}
	var ans struct {
		SDP string `json:"sdp"`
	}
	_ = json.NewDecoder(resp.Body).Decode(&ans)
	resp.Body.Close()
	_ = pc.SetRemoteDescription(webrtc.SessionDescription{Type: webrtc.SDPTypeAnswer, SDP: ans.SDP})

	// Limit: 20 ms frames = 50 packets/s. Within a 3 s window we require 25 silence
	// packets (0.5 s worth), which allows for connection set-up time.
	setMute(true)
	silent := 0
	deadline := time.After(3 * time.Second)
waitSilence:
	for silent < 25 {
		select {
		case p := <-payloads:
			if bytes.Equal(p, silence) {
				silent++
			}
		case <-deadline:
			break waitSilence
		}
	}
	if silent < 25 {
		panic(fmt.Sprintf("FAIL [7b]: only %d Opus silence packets in 3 s while muted (need 25)", silent))
	}
	fmt.Printf("  -> PASS [7b]: WebRTC listener got %d Opus silence packets while muted\n", silent)

	// After unmute, real encoded audio must come back. At 96 kbit/s CBR a 20 ms
	// frame is 240 bytes, far more than the 3-byte silence frame.
	setMute(false)
	deadline = time.After(2 * time.Second)
	for {
		select {
		case p := <-payloads:
			if len(p) > len(silence) {
				fmt.Printf("  -> PASS [7c]: real audio resumed after unmute (%d-byte frames)\n", len(p))
				return
			}
		case <-deadline:
			panic("FAIL [7c]: no real audio within 2 s of unmute")
		}
	}
}
EOF
go run -mod=vendor /tmp/e2e_mute_verify.go

echo ""
echo "======================================================================"
echo "  ALL 7 PRODUCTION TESTS PASSED ON THIS PC"
echo "======================================================================"
