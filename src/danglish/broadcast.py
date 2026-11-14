"""Local Wi-Fi broadcast of translated TTS audio.

The church broadcast server (``broadcast/church``) captures a PulseAudio source
with ffmpeg and fans it out over WebRTC and HTTP MP3.  Danglish plays the English
TTS into a dedicated null sink; the server captures that sink's monitor, so the
broadcast carries the translation rather than the Danish microphone.

The original Ferrum/audio project is copied read-only into this repository; this
module only starts and supervises the copy.
"""

from __future__ import annotations

import contextlib
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .audio.util import resample_audio
from .config import BroadcastConfig
from .errors import DanglishError, DependencyMissingError

_POLL_INTERVAL = 0.5
_START_TIMEOUT = 45.0
_SAMPLE_RATE = 48_000
# A bounded queue decouples the null-sink writer from local speaker playback.
# 16 stereo segments is roughly half a minute of speech even for long sentences.
_AUDIO_QUEUE_SIZE = 16


def _broadcast_dir() -> Path:
    override = os.environ.get("DANGLISH_BROADCAST_DIR")
    if override:
        return Path(override)
    if getattr(sys, "frozen", False):
        bases = []
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bases.append(Path(meipass))
        bases.append(Path(sys.executable).resolve().parent)
        bases.append(Path(sys.executable).resolve().parent / "_internal")
        for base in bases:
            candidate = base / "broadcast" / "church"
            if candidate.is_dir():
                return candidate
    return Path(__file__).resolve().parents[2] / "broadcast" / "church"


def detect_lan_ip() -> str:
    """Return the outbound LAN address without sending any traffic."""

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


class BroadcastSink:
    """A dedicated Pulse null sink that receives the translated TTS audio."""

    def __init__(self, config: BroadcastConfig) -> None:
        self.config = config
        self._pactl = shutil.which("pactl")
        self._pacat = shutil.which("pacat")
        self._module_id: str | None = None
        self._created = False
        self._player: subprocess.Popen[bytes] | None = None
        self._writer: threading.Thread | None = None
        self._audio: queue.Queue[bytes | None] = queue.Queue(maxsize=_AUDIO_QUEUE_SIZE)
        self._error: str | None = None
        self._closed = False

    @property
    def sample_rate(self) -> int:
        return _SAMPLE_RATE

    def _pactl_run(self, *args: str) -> subprocess.CompletedProcess[str]:
        if not self._pactl:
            raise DependencyMissingError(
                "Wi-Fi broadcast needs PulseAudio/PipeWire tools (pactl).\n"
                "  apt: sudo apt install pulseaudio-utils\n"
                "  or use PipeWire's pulse compatibility package."
            )
        return subprocess.run(
            [self._pactl, *args],
            check=True,
            capture_output=True,
            text=True,
        )

    def _sink_exists(self) -> bool:
        result = self._pactl_run("list", "short", "sinks")
        for line in result.stdout.splitlines():
            fields = line.split("\t")
            if len(fields) >= 2 and fields[1] == self.config.sink_name:
                return True
        return False

    def open(self) -> None:
        if not self._sink_exists():
            result = self._pactl_run(
                "load-module",
                "module-null-sink",
                f"sink_name={self.config.sink_name}",
                "rate=48000",
                "channels=2",
                "sink_properties=device.description=Danglish_Broadcast",
            )
            self._module_id = result.stdout.strip() or None
            self._created = True

        if not self._pacat:
            self.close()
            raise DependencyMissingError(
                "Wi-Fi broadcast needs pacat from PulseAudio/PipeWire tools.\n"
                "  apt: sudo apt install pulseaudio-utils\n"
                "  or use PipeWire's pulse compatibility package."
            )

        try:
            self._player = subprocess.Popen(
                [
                    self._pacat,
                    "--playback",
                    "--raw",
                    "--format=float32le",
                    f"--rate={_SAMPLE_RATE}",
                    "--channels=2",
                    f"--device={self.config.sink_name}",
                    "--latency=19200",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError as exc:
            self.close()
            raise DanglishError(
                f"could not open the broadcast sink {self.config.sink_name!r}: {exc}"
            ) from exc

        self._error = None
        self._closed = False
        self._audio = queue.Queue(maxsize=_AUDIO_QUEUE_SIZE)
        self._writer = threading.Thread(
            target=self._writer_loop, name="danglish-broadcast", daemon=True
        )
        self._writer.start()

    def _writer_loop(self) -> None:
        """Write queued PCM to pacat, which paces the sink in real time."""

        player = self._player
        stream = player.stdin if player is not None else None
        while True:
            data = self._audio.get()
            if data is None:
                return
            if self._error is not None or stream is None:
                continue
            try:
                stream.write(data)
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._error = f"broadcast audio write failed: {exc}"

    def write(self, audio: np.ndarray, sample_rate: int) -> None:
        if self._closed or audio.size == 0:
            return
        if self._error is not None:
            raise DanglishError(self._error)
        if self._player is None:
            raise DanglishError("broadcast sink is not open")
        mono = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sample_rate != _SAMPLE_RATE:
            mono = resample_audio(mono, sample_rate, _SAMPLE_RATE)
        data = np.repeat(mono[:, None], 2, axis=1).astype(np.float32, copy=False)
        if self._player.poll() is not None:
            raise DanglishError("broadcast audio player exited (pacat)")
        # Never block the local speaker on the broadcast; the writer thread
        # plays both outputs in parallel.  The bounded queue applies backpressure
        # only if pacat stalls badly.
        self._audio.put(data.tobytes())

    def _discard_queued(self) -> None:
        while True:
            try:
                self._audio.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        self._closed = True
        player = self._player
        self._player = None
        if player is not None:
            # Terminate first: it unblocks a writer mid-pipe and prevents a
            # long shutdown while queued audio plays out.
            with contextlib.suppress(Exception):
                player.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                player.wait(timeout=2)
            if player.poll() is None:
                with contextlib.suppress(Exception):
                    player.kill()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    player.wait(timeout=2)
        writer = self._writer
        self._writer = None
        if writer is not None:
            self._discard_queued()
            with contextlib.suppress(queue.Full):
                self._audio.put_nowait(None)
            writer.join(timeout=2)
        if self._created and self._module_id and self._pactl:
            subprocess.run(
                [self._pactl, "unload-module", self._module_id],
                check=False,
                capture_output=True,
            )
            self._module_id = None
            self._created = False


class BroadcastServer:
    """Supervises the copied church broadcast server."""

    def __init__(self, config: BroadcastConfig, *, directory: Path | None = None) -> None:
        self.config = config
        self.directory = directory or _broadcast_dir()
        self.binary = self.directory / "church-audio-server"
        self.launcher = self.directory / "start_church_broadcast.sh"
        self.process: subprocess.Popen[bytes] | None = None
        self._log_path = Path(tempfile.gettempdir()) / "danglish-broadcast.log"
        self._log = None
        self._host = self._resolve_host()

    @property
    def host(self) -> str:
        return self._host

    def _resolve_host(self) -> str:
        configured = self.config.server_ip.strip()
        if configured and configured != "auto":
            return configured
        return detect_lan_ip() if self.config.local else "10.42.0.1"

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "AUDIO_BACKEND": "pulse",
                "AUDIO_INPUT": self.config.audio_input,
                "SSID": self.config.ssid,
                "SERVER_IP": self._host,
                "HTTP_PORT": str(self.config.http_port),
                "HTTPS_PORT": str(self.config.https_port),
                "WEBRTC_PORT": str(self.config.webrtc_port),
            }
        )
        return env

    def _command(self) -> list[str]:
        if not self.config.local:
            return ["bash", str(self.launcher)]
        return [
            str(self.binary),
            f"--ip={self._host}",
            f"--http-port={self.config.http_port}",
            f"--https-port={self.config.https_port}",
            f"--webrtc-port={self.config.webrtc_port}",
            f"--rtp-port={self.config.rtp_port}",
            "--audio-backend=pulse",
            f"--audio-input={self.config.audio_input}",
            "--print-qr=false",
        ]

    def _build_if_needed(self) -> None:
        if self.binary.is_file():
            return
        if not (self.directory / "main.go").is_file():
            raise DanglishError(
                f"broadcast server missing at {self.binary}; reinstall Danglish "
                "or run from a source checkout"
            )
        go = shutil.which("go")
        if not go:
            raise DanglishError(
                f"broadcast server missing at {self.binary}; install Go or copy "
                "broadcast/church/church-audio-server"
            )
        subprocess.run(
            [go, "build", "-mod=vendor", "-ldflags", "-s -w", "-o", str(self.binary), "main.go"],
            cwd=self.directory,
            check=True,
        )

    def _tail_log(self, lines: int = 12) -> str:
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _wait_ready(self) -> None:
        url = f"http://{self._host}:{self.config.http_port}/stats"
        deadline = time.monotonic() + _START_TIMEOUT
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                tail = self._tail_log()
                raise DanglishError(
                    f"broadcast server exited during startup (code {self.process.returncode}).\n{tail}"
                )
            try:
                with urllib.request.urlopen(url, timeout=1.0) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            time.sleep(_POLL_INTERVAL)
        tail = self._tail_log()
        raise DanglishError(
            f"broadcast server did not become ready at {url} within {_START_TIMEOUT:.0f}s.\n{tail}"
        )

    def start(self) -> None:
        self._host = self._resolve_host()
        self._build_if_needed()
        if not self.config.local and not self.launcher.is_file():
            raise DanglishError(f"broadcast launcher missing at {self.launcher}")

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config.local:
            self._log = self._log_path.open("a", encoding="utf-8")
            stdout = self._log
            stderr = subprocess.STDOUT
        else:
            # Keep sudo's password prompt visible on the terminal.
            stdout = None
            stderr = None

        try:
            self.process = subprocess.Popen(
                self._command(),
                cwd=self.directory,
                env=self._environment(),
                stdin=None,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            self._wait_ready()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                process.wait(timeout=3)
        if self._log is not None:
            self._log.close()
            self._log = None

    def log_path(self) -> Path:
        return self._log_path

    @property
    def listener_url(self) -> str:
        return f"http://{self._host}:{self.config.http_port}/"

    @property
    def speaker_url(self) -> str:
        return f"http://{self._host}:{self.config.http_port}/speaker"

    @property
    def qr_url(self) -> str:
        return f"http://{self._host}:{self.config.http_port}/qr"


class BroadcastSession:
    """A null sink plus the broadcast server, used by ``danglish talk``."""

    def __init__(self, config: BroadcastConfig, *, directory: Path | None = None) -> None:
        self.config = config
        self.sink = BroadcastSink(config)
        self.server = BroadcastServer(config, directory=directory)

    def start(self) -> None:
        if not sys.platform.startswith("linux"):
            raise DanglishError(
                "Wi-Fi broadcast is supported on Linux with PipeWire/PulseAudio only"
            )
        self.sink.open()
        try:
            self.server.start()
        except Exception:
            self.sink.close()
            raise

    def write(self, audio: np.ndarray, sample_rate: int) -> None:
        self.sink.write(audio, sample_rate)

    def stop(self) -> None:
        self.server.stop()
        self.sink.close()

    @property
    def listener_url(self) -> str:
        return self.server.listener_url

    @property
    def speaker_url(self) -> str:
        return self.server.speaker_url

    @property
    def qr_url(self) -> str:
        return self.server.qr_url

    @property
    def log_path(self) -> Path:
        return self.server.log_path()
