package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	_ "embed"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"math/big"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
	"rsc.io/qr"
)

//go:embed index.html
var indexHTML []byte

//go:embed speaker.html
var speakerHTML []byte

// opusSilenceFrame is one 20 ms Opus frame of silence: TOC 0xF8 (config 31 =
// CELT fullband 20 ms, mono, one frame) followed by the 2-byte CELT silence frame.
// 20 ms matches the encoder's -frame_duration 20, so the RTP timestamp step (960
// samples at 48 kHz) stays correct.
var opusSilenceFrame = []byte{0xF8, 0xFF, 0xFE}

const (
	localRTPAddr = "127.0.0.1:5004"
	dscpEF       = 0xb8 // DSCP 46 (EF - Expedited Forwarding) << 2 = 184 (0xB8) for WMM AC_VO queue
)

type ServerConfig struct {
	ServerIP     string
	HTTPPort     int
	HTTPSPort    int
	WebRTCPort   int
	RTPPort      int
	AudioInput   string
	AudioBackend string
	SSID         string
	WiFiPassword string
	PrintQR      bool
}

type Broadcaster struct {
	cfg            ServerConfig
	api            *webrtc.API
	audioTrack     *webrtc.TrackLocalStaticRTP
	activeClients  atomic.Int64
	phoneSpeakerOn atomic.Bool
	muted          atomic.Bool
	bytesReceived  atomic.Uint64
	packetsRecvd   atomic.Uint64
	currentBitrate atomic.Uint64 // in bps
	audioLevelPct  atomic.Int64
	ffmpegRestarts atomic.Uint64
	ffmpegRunning  atomic.Bool
	startTime      time.Time

	rtpMu       sync.Mutex
	seqNum      uint16
	lastSource  uint8 // 1 = ffmpeg, 2 = phone WHIP
	tsBase      uint32
	lastInTS    uint32
	lastOutTS   uint32

	sourceMu      sync.Mutex
	currentInput  string
	ffmpegCancel  context.CancelFunc

	mp3Mu         sync.Mutex
	mp3Listeners  map[int]*net.UDPAddr
	mp3NextID     int
	mp3OutConn    *net.UDPConn
}

type SignalMessage struct {
	Type string `json:"type"`
	SDP  string `json:"sdp"`
}

type StatsResponse struct {
	ActiveListeners    int64   `json:"active_listeners"`
	BitrateKbps        float64 `json:"bitrate_kbps"`
	PacketsTotal       uint64  `json:"packets_total"`
	FFmpegRunning      bool    `json:"ffmpeg_running"`
	FFmpegRestarts     uint64  `json:"ffmpeg_restarts"`
	PhoneSpeakerActive bool    `json:"phone_speaker_active"`
	Muted              bool    `json:"muted"`
	AudioInput         string  `json:"audio_input"`
	AudioLevelPct      int64   `json:"audio_level_pct"`
	UptimeSeconds      int64   `json:"uptime_seconds"`
	ServerIP           string  `json:"server_ip"`
	WebRTCUDPPort      int     `json:"webrtc_udp_port"`
}

type AudioSourceInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

func main() {
	cfg := ServerConfig{}
	flag.StringVar(&cfg.ServerIP, "ip", "10.42.0.1", "Static IP address of the Wi-Fi hotspot interface")
	flag.IntVar(&cfg.HTTPPort, "http-port", 8080, "HTTP portal and listener signaling port")
	flag.IntVar(&cfg.HTTPSPort, "https-port", 8443, "HTTPS port for wireless phone speaker mic (getUserMedia)")
	flag.IntVar(&cfg.WebRTCPort, "webrtc-port", 8555, "Single UDP mux port for all WebRTC audio traffic")
	flag.IntVar(&cfg.RTPPort, "rtp-port", 5004, "Local UDP port for FFmpeg RTP ingest")
	flag.StringVar(&cfg.AudioInput, "audio-input", "default", "Audio capture device name (e.g., 'default')")
	flag.StringVar(&cfg.AudioBackend, "audio-backend", "pulse", "Audio capture backend: 'pulse' (PipeWire/PulseAudio) or 'alsa'")
	flag.StringVar(&cfg.SSID, "ssid", "Church_Live_Audio", "Wi-Fi SSID for QR display")
	flag.StringVar(&cfg.WiFiPassword, "wifi-password", "", "Wi-Fi password for QR display (empty = open network, no password)")
	flag.BoolVar(&cfg.PrintQR, "print-qr", true, "Print scannable UTF-8 QR codes to terminal on startup")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	broadcaster, udpConn, err := NewBroadcaster(cfg)
	if err != nil {
		log.Fatalf("[FATAL] Failed to initialize WebRTC broadcaster: %v", err)
	}
	defer udpConn.Close()

	if cfg.PrintQR {
		broadcaster.printTerminalQRCodes()
	}

	// 1. Start RTP ingest loop (127.0.0.1:5004 -> shared TrackLocalStaticRTP)
	go broadcaster.runRTPIngestLoop(ctx)

	// 2. Start bitrate measurement ticker
	go broadcaster.runBitrateMonitor(ctx)

	// 3. Start supervised FFmpeg low-latency Opus encoder
	go broadcaster.superviseFFmpeg(ctx)

	// 4. Configure HTTP/HTTPS routes
	mux := http.NewServeMux()
	mux.HandleFunc("/", broadcaster.handleIndex)
	mux.HandleFunc("/speaker", broadcaster.handleSpeaker)
	mux.HandleFunc("/whep", broadcaster.handleOffer)
	mux.HandleFunc("/offer", broadcaster.handleOffer)
	mux.HandleFunc("/whip", broadcaster.handleWHIP)
	mux.HandleFunc("/stream.mp3", broadcaster.handleMP3Fallback)
	mux.HandleFunc("/stats", broadcaster.handleStats)
	mux.HandleFunc("/qr", broadcaster.handleQR)
	mux.HandleFunc("/api/sources", broadcaster.handleListSources)
	mux.HandleFunc("/api/source", broadcaster.handleSetSource)
	mux.HandleFunc("/api/mute", broadcaster.handleMute)

	// OS Connectivity & Captive Portal Probe Endpoints (keeps iPhones & Androids locked onto Wi-Fi)
	mux.HandleFunc("/hotspot-detect.html", broadcaster.handleAppleHotspotProbe)
	mux.HandleFunc("/library/test/success.html", broadcaster.handleAppleHotspotProbe)
	mux.HandleFunc("/generate_204", broadcaster.handleAndroidProbe)
	mux.HandleFunc("/gen_204", broadcaster.handleAndroidProbe)
	mux.HandleFunc("/ncsi.txt", broadcaster.handleNCSIProbe)
	mux.HandleFunc("/connecttest.txt", broadcaster.handleConnectTestProbe)

	httpServer := &http.Server{
		Addr:              fmt.Sprintf("0.0.0.0:%d", cfg.HTTPPort),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	http80Server := &http.Server{
		Addr:              "0.0.0.0:80",
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	tlsCert, err := generateSelfSignedCert(cfg.ServerIP)
	if err != nil {
		log.Fatalf("[FATAL] Failed to generate self-signed TLS cert: %v", err)
	}
	httpsServer := &http.Server{
		Addr:              fmt.Sprintf("0.0.0.0:%d", cfg.HTTPSPort),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		TLSConfig: &tls.Config{
			Certificates: []tls.Certificate{tlsCert},
		},
	}

	go func() {
		log.Printf("[INFO] Listener Portal (HTTP)   : http://%s:%d (and http://%s)", cfg.ServerIP, cfg.HTTPPort, cfg.ServerIP)
		log.Printf("[INFO] Speaker Console (HTTPS)  : https://%s:%d/speaker", cfg.ServerIP, cfg.HTTPSPort)
		log.Printf("[INFO] Speaker Console (HTTP)   : http://%s:%d/speaker", cfg.ServerIP, cfg.HTTPPort)
		log.Printf("[INFO] Operator QR Dashboard    : http://%s:%d/qr", cfg.ServerIP, cfg.HTTPPort)
		log.Printf("[INFO] WebRTC Single UDP Port   : %s:%d (DSCP EF 0xB8)", cfg.ServerIP, cfg.WebRTCPort)
		if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("[FATAL] HTTP server failed: %v", err)
		}
	}()

	if cfg.HTTPPort != 80 {
		go func() {
			if err := http80Server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Printf("[INFO] Port 80 listener not bound (%v); using port %d", err, cfg.HTTPPort)
			} else {
				log.Printf("[INFO] Port 80 HTTP listener active on http://%s", cfg.ServerIP)
			}
		}()
	}

	go func() {
		if err := httpsServer.ListenAndServeTLS("", ""); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("[WARN] HTTPS server failed: %v", err)
		}
	}()

	<-ctx.Done()
	log.Println("[INFO] Shutting down broadcast server...")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_ = httpServer.Shutdown(shutdownCtx)
	_ = http80Server.Shutdown(shutdownCtx)
	_ = httpsServer.Shutdown(shutdownCtx)
}

func NewBroadcaster(cfg ServerConfig) (*Broadcaster, *net.UDPConn, error) {
	mediaEngine := &webrtc.MediaEngine{}
	opusCapability := webrtc.RTPCodecCapability{
		MimeType:     webrtc.MimeTypeOpus,
		ClockRate:    48000,
		Channels:     2,
		SDPFmtpLine:  "minptime=10;useinbandfec=1;stereo=0;sprop-stereo=0",
		RTCPFeedback: nil,
	}
	if err := mediaEngine.RegisterCodec(webrtc.RTPCodecParameters{
		RTPCodecCapability: opusCapability,
		PayloadType:        111,
	}, webrtc.RTPCodecTypeAudio); err != nil {
		return nil, nil, fmt.Errorf("register opus codec: %w", err)
	}

	udpAddr := &net.UDPAddr{
		IP:   net.ParseIP("0.0.0.0"),
		Port: cfg.WebRTCPort,
	}
	udpConn, err := net.ListenUDP("udp4", udpAddr)
	if err != nil {
		return nil, nil, fmt.Errorf("listen WebRTC UDP port %d: %w", cfg.WebRTCPort, err)
	}

	if err := setSocketDSCPEF(udpConn); err != nil {
		log.Printf("[WARN] Could not set DSCP EF (0xB8) on UDP socket: %v", err)
	} else {
		log.Printf("[INFO] Set IP_TOS=0xB8 (DSCP EF / WMM Voice) on UDP port %d", cfg.WebRTCPort)
	}

	_ = udpConn.SetWriteBuffer(4 * 1024 * 1024)

	settingEngine := webrtc.SettingEngine{}
	settingEngine.SetNetworkTypes([]webrtc.NetworkType{webrtc.NetworkTypeUDP4})
	settingEngine.SetICEUDPMux(webrtc.NewICEUDPMux(nil, udpConn))
	settingEngine.SetNAT1To1IPs([]string{cfg.ServerIP}, webrtc.ICECandidateTypeHost)
	settingEngine.SetIncludeLoopbackCandidate(true)

	api := webrtc.NewAPI(
		webrtc.WithMediaEngine(mediaEngine),
		webrtc.WithSettingEngine(settingEngine),
	)

	audioTrack, err := webrtc.NewTrackLocalStaticRTP(
		opusCapability,
		"church-live-audio",
		"church-main-stream",
	)
	if err != nil {
		udpConn.Close()
		return nil, nil, fmt.Errorf("create shared audio track: %w", err)
	}

	mp3OutConn, _ := net.ListenUDP("udp4", &net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 0})

	return &Broadcaster{
		cfg:          cfg,
		api:          api,
		audioTrack:   audioTrack,
		currentInput: cfg.AudioInput,
		startTime:    time.Now(),
		mp3Listeners: make(map[int]*net.UDPAddr),
		mp3OutConn:   mp3OutConn,
	}, udpConn, nil
}

func setSocketDSCPEF(conn *net.UDPConn) error {
	rawConn, err := conn.SyscallConn()
	if err != nil {
		return err
	}
	var sockErr error
	err = rawConn.Control(func(fd uintptr) {
		sockErr = syscall.SetsockoptInt(int(fd), syscall.IPPROTO_IP, syscall.IP_TOS, dscpEF)
	})
	if err != nil {
		return err
	}
	return sockErr
}

// writeAudioRTP normalizes RTP sequence numbers and timestamps across source switches
// (e.g. switching between Laptop FFmpeg mic and Speaker Phone WHIP mic) and writes
// to the shared TrackLocalStaticRTP.
func (b *Broadcaster) writeAudioRTP(raw []byte, sourceID uint8) {
	var pkt rtp.Packet
	if err := pkt.Unmarshal(raw); err != nil {
		return
	}

	// While the laptop mic is muted, keep sending packets but replace the audio with
	// an Opus silence frame. Dropping the packets instead would starve every
	// /stream.mp3 listener's ffmpeg (no input -> no MP3 bytes -> the phone's player
	// stalls and does not come back), and leaves WebRTC jitter buffers empty.
	mutedNow := b.muted.Load() && sourceID == 1
	if mutedNow {
		pkt.Payload = opusSilenceFrame
	}

	b.bytesReceived.Add(uint64(len(raw)))
	b.packetsRecvd.Add(1)

	// Estimate voice activity level from Opus payload length & TOC byte
	payloadLen := len(pkt.Payload)
	level := int64(0)
	if !mutedNow && payloadLen > 6 {
		level = int64((payloadLen - 6) * 100 / 90)
		if level > 100 {
			level = 100
		}
	}
	b.audioLevelPct.Store(level)

	b.rtpMu.Lock()
	b.seqNum++
	pkt.SequenceNumber = b.seqNum
	if b.lastSource != sourceID {
		b.lastSource = sourceID
		b.tsBase = b.lastOutTS + 960 - pkt.Timestamp
	}
	pkt.Timestamp += b.tsBase
	b.lastOutTS = pkt.Timestamp
	b.rtpMu.Unlock()

	if err := b.audioTrack.WriteRTP(&pkt); err != nil && !errors.Is(err, io.ErrClosedPipe) {
		log.Printf("[WARN] RTP track write error: %v", err)
	}

	// Also forward normalized RTP packet to any active legacy HTTP MP3 fallback streams
	b.mp3Mu.Lock()
	if len(b.mp3Listeners) > 0 && b.mp3OutConn != nil {
		if outBytes, err := pkt.Marshal(); err == nil {
			for _, addr := range b.mp3Listeners {
				_, _ = b.mp3OutConn.WriteToUDP(outBytes, addr)
			}
		}
	}
	b.mp3Mu.Unlock()
}

func (b *Broadcaster) runRTPIngestLoop(ctx context.Context) {
	ingestAddr := fmt.Sprintf("127.0.0.1:%d", b.cfg.RTPPort)
	addr, err := net.ResolveUDPAddr("udp4", ingestAddr)
	if err != nil {
		log.Fatalf("[FATAL] Resolve RTP address %s: %v", ingestAddr, err)
	}
	conn, err := net.ListenUDP("udp4", addr)
	if err != nil {
		log.Fatalf("[FATAL] Listen RTP ingest %s: %v", ingestAddr, err)
	}
	defer conn.Close()
	_ = conn.SetReadBuffer(1024 * 1024)

	go func() {
		<-ctx.Done()
		_ = conn.Close()
	}()

	log.Printf("[INFO] RTP Ingest listening on udp://%s", ingestAddr)

	buf := make([]byte, 1500)
	for {
		n, _, err := conn.ReadFromUDP(buf)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			continue
		}
		if n < 12 {
			continue
		}
		// If a phone speaker is actively streaming via WHIP, ignore laptop mic packets
		if b.phoneSpeakerOn.Load() {
			continue
		}
		b.writeAudioRTP(buf[:n], 1)
	}
}

func (b *Broadcaster) runBitrateMonitor(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	var lastBytes uint64
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			curr := b.bytesReceived.Load()
			delta := curr - lastBytes
			lastBytes = curr
			b.currentBitrate.Store(delta * 8)
		}
	}
}

func (b *Broadcaster) superviseFFmpeg(ctx context.Context) {
	backend := b.cfg.AudioBackend
	for {
		if ctx.Err() != nil {
			return
		}

		b.sourceMu.Lock()
		inputDevice := b.currentInput
		ffCtx, ffCancel := context.WithCancel(ctx)
		b.ffmpegCancel = ffCancel
		b.sourceMu.Unlock()

		args := []string{
			"-nostdin",
			"-hide_banner",
			"-loglevel", "error",
			"-fflags", "nobuffer",
			"-flags", "low_delay",
			"-f", backend,
		}
		if backend == "pulse" {
			// 1920-byte PulseAudio fragment size eliminates 85ms PipeWire buffer bursts,
			// pacing 20ms Opus RTP packets at a steady ~20.0ms interval (0 gaps > 35ms).
			args = append(args, "-fragment_size", "1920")
		}
		args = append(args,
			"-i", inputDevice,
			"-af", "asetpts=N/SR/TB",
			"-ac", "1",
			"-ar", "48000",
			"-c:a", "libopus",
			"-b:a", "96k",
			"-vbr", "off",
			"-application", "audio",
			"-frame_duration", "20",
			"-payload_type", "111",
			"-f", "rtp",
			fmt.Sprintf("rtp://127.0.0.1:%d", b.cfg.RTPPort),
		)

		log.Printf("[INFO] Starting FFmpeg encoder (backend=%s, input=%s)", backend, inputDevice)

		start := time.Now()
		cmd := exec.CommandContext(ffCtx, "ffmpeg", args...)
		cmd.Stdout = io.Discard
		cmd.Stderr = os.Stderr

		b.ffmpegRunning.Store(true)
		err := cmd.Run()
		b.ffmpegRunning.Store(false)
		ffCancel()

		if ctx.Err() != nil {
			return
		}

		b.ffmpegRestarts.Add(1)
		runDuration := time.Since(start)

		if runDuration < 2*time.Second && err != nil {
			if backend == "pulse" {
				log.Println("[WARN] 'pulse' backend exited quickly; trying fallback to 'alsa'...")
				backend = "alsa"
			} else if backend == "alsa" && b.cfg.AudioBackend == "pulse" {
				backend = "pulse"
			}
		}

		select {
		case <-ctx.Done():
			return
		case <-time.After(600 * time.Millisecond):
		}
	}
}

func (b *Broadcaster) handleIndex(w http.ResponseWriter, r *http.Request) {
	// Handle Apple / Android connectivity checks if they hit root with captive hostnames
	host := strings.ToLower(r.Host)
	if strings.Contains(host, "captive.apple.com") {
		b.handleAppleHotspotProbe(w, r)
		return
	}
	if strings.Contains(host, "connectivitycheck.gstatic.com") || strings.Contains(host, "clients3.google.com") {
		b.handleAndroidProbe(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store, no-cache, must-revalidate")
	_, _ = w.Write(indexHTML)
}

func (b *Broadcaster) handleAppleHotspotProbe(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>"))
}

func (b *Broadcaster) handleAndroidProbe(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(http.StatusNoContent)
}

func (b *Broadcaster) handleNCSIProbe(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("Microsoft NCSI"))
}

func (b *Broadcaster) handleConnectTestProbe(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("Microsoft Connect Test"))
}

// handleMP3Fallback provides a universal live HTTP MP3 audio stream for very old
// iPhones/Androids that lack modern WebRTC support, fed from the live RTP fan-out
// (works with both Laptop Mic and Speaker Phone Wireless Mic).
func (b *Broadcaster) handleMP3Fallback(w http.ResponseWriter, r *http.Request) {
	probeConn, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: 0})
	if err != nil {
		http.Error(w, "Failed to allocate RTP port", http.StatusInternalServerError)
		return
	}
	rtpPort := probeConn.LocalAddr().(*net.UDPAddr).Port
	_ = probeConn.Close()

	sdpContent := fmt.Sprintf("v=0\no=- 0 0 IN IP4 127.0.0.1\ns=Live\nc=IN IP4 127.0.0.1\nt=0 0\nm=audio %d RTP/AVP 111\na=rtpmap:111 opus/48000/2\n", rtpPort)
	sdpFile, err := os.CreateTemp("", "church-mp3-*.sdp")
	if err != nil {
		http.Error(w, "Failed to create SDP", http.StatusInternalServerError)
		return
	}
	sdpPath := sdpFile.Name()
	_, _ = sdpFile.WriteString(sdpContent)
	_ = sdpFile.Close()
	defer os.Remove(sdpPath)

	targetAddr := &net.UDPAddr{IP: net.ParseIP("127.0.0.1"), Port: rtpPort}
	b.mp3Mu.Lock()
	b.mp3NextID++
	id := b.mp3NextID
	b.mp3Listeners[id] = targetAddr
	b.mp3Mu.Unlock()

	defer func() {
		b.mp3Mu.Lock()
		delete(b.mp3Listeners, id)
		b.mp3Mu.Unlock()
	}()

	w.Header().Set("Content-Type", "audio/mpeg")
	w.Header().Set("Cache-Control", "no-store, no-cache, must-revalidate")
	w.Header().Set("Connection", "keep-alive")

	args := []string{
		"-nostdin",
		"-hide_banner",
		"-loglevel", "error",
		"-protocol_whitelist", "file,udp,rtp",
		"-fflags", "+nobuffer+flush_packets",
		"-flags", "low_delay",
		"-flush_packets", "1",
		"-i", sdpPath,
		"-ac", "1",
		"-ar", "44100",
		"-c:a", "libmp3lame",
		"-b:a", "96k",
		"-reservoir", "0",
		"-write_xing", "0",
		"-f", "mp3",
		"pipe:1",
	}

	cmd := exec.CommandContext(r.Context(), "ffmpeg", args...)
	cmd.Stdout = w
	cmd.Stderr = io.Discard

	b.activeClients.Add(1)
	defer b.activeClients.Add(-1)

	_ = cmd.Run()
}

func (b *Broadcaster) handleSpeaker(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store, no-cache, must-revalidate")
	_, _ = w.Write(speakerHTML)
}

// handleWHIP accepts an upstream WebRTC audio stream from the Speaker's phone microphone
// and fans it out to all connected Listeners.
func (b *Broadcaster) handleWHIP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	var msg SignalMessage
	if err := json.NewDecoder(io.LimitReader(r.Body, 64*1024)).Decode(&msg); err != nil {
		http.Error(w, "Invalid JSON offer", http.StatusBadRequest)
		return
	}

	peerConnection, err := b.api.NewPeerConnection(webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{},
	})
	if err != nil {
		http.Error(w, "Failed to create PeerConnection", http.StatusInternalServerError)
		return
	}

	peerConnection.OnTrack(func(track *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
		log.Printf("[INFO] Speaker Phone wireless microphone connected (%s)", track.Codec().MimeType)
		b.phoneSpeakerOn.Store(true)
		defer b.phoneSpeakerOn.Store(false)

		buf := make([]byte, 1500)
		for {
			n, _, readErr := track.Read(buf)
			if readErr != nil {
				log.Printf("[INFO] Speaker Phone wireless microphone disconnected")
				return
			}
			if n >= 12 {
				b.writeAudioRTP(buf[:n], 2)
			}
		}
	})

	peerConnection.OnICEConnectionStateChange(func(state webrtc.ICEConnectionState) {
		if state == webrtc.ICEConnectionStateDisconnected ||
			state == webrtc.ICEConnectionStateFailed ||
			state == webrtc.ICEConnectionStateClosed {
			b.phoneSpeakerOn.Store(false)
			_ = peerConnection.Close()
		}
	})

	if err := peerConnection.SetRemoteDescription(webrtc.SessionDescription{
		Type: webrtc.SDPTypeOffer,
		SDP:  msg.SDP,
	}); err != nil {
		_ = peerConnection.Close()
		http.Error(w, "Invalid remote SDP", http.StatusBadRequest)
		return
	}

	answer, err := peerConnection.CreateAnswer(nil)
	if err != nil {
		_ = peerConnection.Close()
		http.Error(w, "Failed to create SDP answer", http.StatusInternalServerError)
		return
	}

	gatherComplete := webrtc.GatheringCompletePromise(peerConnection)
	if err := peerConnection.SetLocalDescription(answer); err != nil {
		_ = peerConnection.Close()
		http.Error(w, "Failed to set local SDP", http.StatusInternalServerError)
		return
	}

	select {
	case <-gatherComplete:
	case <-time.After(3 * time.Second):
	}

	localDesc := peerConnection.LocalDescription()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(SignalMessage{
		Type: localDesc.Type.String(),
		SDP:  localDesc.SDP,
	})
}

func (b *Broadcaster) handleOffer(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	var offer webrtc.SessionDescription
	contentType := r.Header.Get("Content-Type")
	bodyBytes, err := io.ReadAll(io.LimitReader(r.Body, 64*1024))
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusBadRequest)
		return
	}

	if strings.Contains(contentType, "application/sdp") {
		offer = webrtc.SessionDescription{
			Type: webrtc.SDPTypeOffer,
			SDP:  string(bodyBytes),
		}
	} else {
		var msg SignalMessage
		if err := json.Unmarshal(bodyBytes, &msg); err != nil {
			http.Error(w, "Invalid JSON offer payload", http.StatusBadRequest)
			return
		}
		offer = webrtc.SessionDescription{
			Type: webrtc.SDPTypeOffer,
			SDP:  msg.SDP,
		}
	}

	peerConnection, err := b.api.NewPeerConnection(webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{},
	})
	if err != nil {
		http.Error(w, "Failed to create PeerConnection", http.StatusInternalServerError)
		return
	}

	rtpSender, err := peerConnection.AddTrack(b.audioTrack)
	if err != nil {
		_ = peerConnection.Close()
		http.Error(w, "Failed to attach audio track", http.StatusInternalServerError)
		return
	}

	go func() {
		rtcpBuf := make([]byte, 1500)
		for {
			if _, _, rtcpErr := rtpSender.Read(rtcpBuf); rtcpErr != nil {
				return
			}
		}
	}()

	var counted atomic.Bool
	peerConnection.OnICEConnectionStateChange(func(state webrtc.ICEConnectionState) {
		switch state {
		case webrtc.ICEConnectionStateConnected, webrtc.ICEConnectionStateCompleted:
			if counted.CompareAndSwap(false, true) {
				active := b.activeClients.Add(1)
				log.Printf("[INFO] Listener connected (active listeners: %d)", active)
			}
		case webrtc.ICEConnectionStateDisconnected, webrtc.ICEConnectionStateFailed, webrtc.ICEConnectionStateClosed:
			if counted.CompareAndSwap(true, false) {
				active := b.activeClients.Add(-1)
				log.Printf("[INFO] Listener disconnected (active listeners: %d)", active)
			}
			if state == webrtc.ICEConnectionStateFailed || state == webrtc.ICEConnectionStateClosed {
				_ = peerConnection.Close()
			}
		}
	})

	if err := peerConnection.SetRemoteDescription(offer); err != nil {
		_ = peerConnection.Close()
		http.Error(w, "Invalid remote SDP", http.StatusBadRequest)
		return
	}

	answer, err := peerConnection.CreateAnswer(nil)
	if err != nil {
		_ = peerConnection.Close()
		http.Error(w, "Failed to create SDP answer", http.StatusInternalServerError)
		return
	}

	gatherComplete := webrtc.GatheringCompletePromise(peerConnection)
	if err := peerConnection.SetLocalDescription(answer); err != nil {
		_ = peerConnection.Close()
		http.Error(w, "Failed to set local SDP", http.StatusInternalServerError)
		return
	}

	select {
	case <-gatherComplete:
	case <-time.After(3 * time.Second):
	}

	localDesc := peerConnection.LocalDescription()
	if strings.Contains(contentType, "application/sdp") {
		w.Header().Set("Content-Type", "application/sdp")
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(localDesc.SDP))
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(SignalMessage{
		Type: localDesc.Type.String(),
		SDP:  localDesc.SDP,
	})
}

func (b *Broadcaster) handleListSources(w http.ResponseWriter, r *http.Request) {
	sources := []AudioSourceInfo{
		{Name: "default", Description: "Default System Microphone"},
	}
	out, err := exec.Command("pactl", "list", "short", "sources").Output()
	if err == nil {
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		for _, line := range lines {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				name := fields[1]
				label := name
				if strings.HasSuffix(name, ".monitor") {
					label = "System Playback Monitor (" + strings.TrimSuffix(name, ".monitor") + ")"
				} else if strings.Contains(name, "usb-") {
					label = "USB Audio Interface (" + name + ")"
				} else if strings.Contains(name, "Mic") {
					label = "Built-in Laptop Mic (" + name + ")"
				}
				sources = append(sources, AudioSourceInfo{Name: name, Description: label})
			}
		}
	}

	b.sourceMu.Lock()
	curr := b.currentInput
	b.sourceMu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"current": curr,
		"sources": sources,
	})
}

func (b *Broadcaster) handleSetSource(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Source string `json:"source"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Source == "" {
		http.Error(w, "Invalid source", http.StatusBadRequest)
		return
	}

	b.sourceMu.Lock()
	b.currentInput = req.Source
	if b.ffmpegCancel != nil {
		b.ffmpegCancel()
	}
	b.sourceMu.Unlock()

	log.Printf("[INFO] Switched laptop audio input source to: %s", req.Source)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok", "source": req.Source})
}

func (b *Broadcaster) handleMute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Muted bool `json:"muted"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid payload", http.StatusBadRequest)
		return
	}
	b.muted.Store(req.Muted)
	log.Printf("[INFO] Broadcast mute state set to: %v", req.Muted)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]bool{"muted": req.Muted})
}

func (b *Broadcaster) handleStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")

	b.sourceMu.Lock()
	currInput := b.currentInput
	b.sourceMu.Unlock()

	resp := StatsResponse{
		ActiveListeners:    b.activeClients.Load(),
		BitrateKbps:        float64(b.currentBitrate.Load()) / 1000.0,
		PacketsTotal:       b.packetsRecvd.Load(),
		FFmpegRunning:      b.ffmpegRunning.Load(),
		FFmpegRestarts:     b.ffmpegRestarts.Load(),
		PhoneSpeakerActive: b.phoneSpeakerOn.Load(),
		Muted:              b.muted.Load(),
		AudioInput:         currInput,
		AudioLevelPct:      b.audioLevelPct.Load(),
		UptimeSeconds:      int64(time.Since(b.startTime).Seconds()),
		ServerIP:           b.cfg.ServerIP,
		WebRTCUDPPort:      b.cfg.WebRTCPort,
	}
	_ = json.NewEncoder(w).Encode(resp)
}

func (b *Broadcaster) handleQR(w http.ResponseWriter, r *http.Request) {
	wifiPayload := b.wifiQRPayload()
	listenerURL := fmt.Sprintf("http://%s:%d", b.cfg.ServerIP, b.cfg.HTTPPort)
	speakerURL := fmt.Sprintf("https://%s:%d/speaker", b.cfg.ServerIP, b.cfg.HTTPSPort)

	wifiSVG := generateQRCodeSVG(wifiPayload)
	listenerSVG := generateQRCodeSVG(listenerURL)
	speakerSVG := generateQRCodeSVG(speakerURL)

	if r.URL.Query().Get("raw") == "1" {
		w.Header().Set("Content-Type", "image/svg+xml")
		_, _ = w.Write([]byte(listenerSVG))
		return
	}

	page := fmt.Sprintf(`<!DOCTYPE html>
<html lang="da">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Holstebro Pinsekirke</title>
<style>
  body { background:radial-gradient(circle at 50%% 0%%, #211b15 0%%, #0e0c0a 75%%); color:#faf6f0; font-family:Georgia,"Times New Roman",serif; margin:0; padding:32px 20px; min-height:100vh; display:flex; flex-direction:column; align-items:center; justify-content:center; box-sizing:border-box; }
  .cross { width:34px; height:46px; stroke:#d4af37; stroke-width:2.2; fill:none; margin-bottom:10px; }
  h1 { margin:0 0 28px; font-size:2.1rem; font-weight:400; letter-spacing:0.04em; color:#f3e5ab; text-align:center; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:24px; width:100%%; max-width:1040px; }
  .card { background:#181411; border:1px solid #2e261f; border-radius:26px; padding:28px 22px; text-align:center; box-shadow:0 20px 50px rgba(0,0,0,0.7); display:flex; flex-direction:column; align-items:center; }
  .card h2 { margin:0 0 18px; font-size:1.35rem; font-weight:400; color:#d4af37; letter-spacing:0.05em; text-transform:uppercase; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
  .qr-box { background:#ffffff; padding:14px; border-radius:18px; display:inline-block; }
  .qr-box svg { width:220px; height:220px; display:block; }
  .code { margin-top:16px; font-family:monospace; font-size:0.95rem; color:#8c7f70; word-break:break-all; }
</style>
</head>
<body>
  <svg class="cross" viewBox="0 0 24 32">
    <line x1="12" y1="2" x2="12" y2="30" stroke-linecap="round"/>
    <line x1="4" y1="10" x2="20" y2="10" stroke-linecap="round"/>
  </svg>
  <h1>Holstebro Pinsekirke</h1>
  <div class="grid">
    <div class="card">
      <h2>1. Wi-Fi</h2>
      <div class="qr-box">%s</div>
      <div class="code">%s</div>
    </div>
    <div class="card">
      <h2>2. Listen</h2>
      <div class="qr-box">%s</div>
      <div class="code">%s</div>
    </div>
    <div class="card">
      <h2>3. Speak</h2>
      <div class="qr-box">%s</div>
      <div class="code">%s</div>
    </div>
  </div>
</body>
</html>`, wifiSVG, b.cfg.SSID, listenerSVG, listenerURL, speakerSVG, speakerURL)

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(page))
}

// wifiQRPayload returns the standard Wi-Fi join QR payload. With no password the
// network is open, and phones expect T:nopass (a WPA entry with an empty password
// makes some phone cameras refuse to join).
func (b *Broadcaster) wifiQRPayload() string {
	if b.cfg.WiFiPassword == "" {
		return fmt.Sprintf("WIFI:T:nopass;S:%s;;", qrEscape(b.cfg.SSID))
	}
	return fmt.Sprintf("WIFI:T:WPA;S:%s;P:%s;;", qrEscape(b.cfg.SSID), qrEscape(b.cfg.WiFiPassword))
}

// qrEscape escapes the characters that are special in a WIFI: QR payload.
func qrEscape(s string) string {
	r := strings.NewReplacer(`\`, `\\`, `;`, `\;`, `,`, `\,`, `:`, `\:`, `"`, `\"`)
	return r.Replace(s)
}

func (b *Broadcaster) printTerminalQRCodes() {
	wifiPayload := b.wifiQRPayload()
	listenerURL := fmt.Sprintf("http://%s:%d", b.cfg.ServerIP, b.cfg.HTTPPort)
	speakerURL := fmt.Sprintf("https://%s:%d/speaker", b.cfg.ServerIP, b.cfg.HTTPSPort)

	fmt.Println()
	fmt.Println("======================================================================")
	fmt.Println("  [1/3] WI-FI HOTSPOT QR CODE (Scan first on Speaker & Listener phones)")
	if b.cfg.WiFiPassword == "" {
		fmt.Printf("        SSID: %s   |   Open network (no password)\n", b.cfg.SSID)
	} else {
		fmt.Printf("        SSID: %s   |   Password: %s\n", b.cfg.SSID, b.cfg.WiFiPassword)
	}
	fmt.Println("======================================================================")
	fmt.Print(generateQRCodeTerminal(wifiPayload))

	fmt.Println("======================================================================")
	fmt.Println("  [2/3] LISTENER QR CODE (Congregation scans to hear live audio)")
	fmt.Printf("        URL: %s\n", listenerURL)
	fmt.Println("======================================================================")
	fmt.Print(generateQRCodeTerminal(listenerURL))

	fmt.Println("======================================================================")
	fmt.Println("  [3/3] SPEAKER / PREACHER QR CODE (Wireless Phone Mic & Mic Control)")
	fmt.Printf("        HTTPS (Phone Mic): %s\n", speakerURL)
	fmt.Printf("        HTTP  (Control)  : %s/speaker\n", listenerURL)
	fmt.Println("======================================================================")
	fmt.Print(generateQRCodeTerminal(speakerURL))
	fmt.Println("======================================================================")
	fmt.Printf("  Full-Screen Browser QR Page: %s/qr\n", listenerURL)
	fmt.Println("======================================================================")
	fmt.Println()
}

func generateSelfSignedCert(serverIP string) (tls.Certificate, error) {
	priv, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return tls.Certificate{}, err
	}
	serial, err := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	if err != nil {
		return tls.Certificate{}, err
	}
	tmpl := x509.Certificate{
		SerialNumber: serial,
		Subject: pkix.Name{
			CommonName:   serverIP,
			Organization: []string{"Sanctuary Live Audio"},
		},
		NotBefore:             time.Now().Add(-1 * time.Hour),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour),
		KeyUsage:              x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		DNSNames:              []string{"audio.lan", "localhost"},
		IPAddresses:           []net.IP{net.ParseIP("127.0.0.1")},
	}
	if ip := net.ParseIP(serverIP); ip != nil {
		tmpl.IPAddresses = append(tmpl.IPAddresses, ip)
	}
	derBytes, err := x509.CreateCertificate(rand.Reader, &tmpl, &tmpl, &priv.PublicKey, priv)
	if err != nil {
		return tls.Certificate{}, err
	}
	return tls.Certificate{
		Certificate: [][]byte{derBytes},
		PrivateKey:  priv,
	}, nil
}

// ----------------------------------------------------------------------------
// Pure-Go ISO/IEC 18004 QR Code Generator (Versions 2-L, 3-L, 4-L)
// Supports payloads up to 78 bytes in 8-bit Byte Mode (SVG + UTF-8 Terminal).
// ----------------------------------------------------------------------------

func generateQRCodeSVG(text string) string {
	matrix := encodeQR(text)
	size := len(matrix)
	const quiet = 3
	total := size + quiet*2

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="240" height="240" shape-rendering="crispEdges">`, total, total))
	sb.WriteString(fmt.Sprintf(`<rect width="%d" height="%d" fill="#ffffff"/>`, total, total))
	sb.WriteString(`<path fill="#000000" d="`)
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if matrix[r][c] {
				sb.WriteString(fmt.Sprintf("M%d,%dh1v1h-1z", c+quiet, r+quiet))
			}
		}
	}
	sb.WriteString(`"/></svg>`)
	return sb.String()
}

func generateQRCodeTerminal(text string) string {
	matrix := encodeQR(text)
	size := len(matrix)
	const quiet = 2
	total := size + quiet*2

	isDark := func(r, c int) bool {
		rr, cc := r-quiet, c-quiet
		if rr < 0 || rr >= size || cc < 0 || cc >= size {
			return false // white quiet zone
		}
		return matrix[rr][cc]
	}

	var sb strings.Builder
	for r := 0; r < total; r += 2 {
		sb.WriteString("  ")
		for c := 0; c < total; c++ {
			top := isDark(r, c)
			bot := false
			if r+1 < total {
				bot = isDark(r+1, c)
			}
			// Invert for dark terminal background (white = full block, black = space)
			switch {
			case !top && !bot:
				sb.WriteString("█")
			case !top && bot:
				sb.WriteString("▀")
			case top && !bot:
				sb.WriteString("▄")
			default:
				sb.WriteString(" ")
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func encodeQR(text string) [][]bool {
	code, err := qr.Encode(text, qr.M)
	if err != nil {
		return [][]bool{{true}}
	}
	size := code.Size
	mat := make([][]bool, size)
	for r := 0; r < size; r++ {
		mat[r] = make([]bool, size)
		for c := 0; c < size; c++ {
			mat[r][c] = code.Black(c, r)
		}
	}
	return mat
}
