"""Microphone-to-speaker loop: capture → VAD → ASR → MT → TTS → playback."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from typing import Any

import numpy as np

from ..backends.base import TranslatorBackend
from ..config import AppConfig, BroadcastConfig
from ..context import ContextProfile, get_profile
from ..errors import DanglishError, DependencyMissingError
from ..memory import current_rss_mb
from ..speech import SpeechTranslator, SpeechUtterance
from ..tts import create_tts
from .asr import DEFAULT_ASR_MODEL, WhisperASR
from .device import (
    channel_index,
    choose_input_channels,
    choose_output_channels,
    choose_rate,
    describe_device,
    resolve_device,
)
from .runtime import TalkRuntime
from .util import resample_audio
from .vad import EnergyVAD

_SAMPLE_RATE = 16_000
_FRAME_MS = 30


def run_talk(
    backend: TranslatorBackend,
    config: AppConfig,
    *,
    asr_model: str | None = None,
    asr_compute_type: str = "int8",
    tts_voice: str | None = None,
    tts_backend: str | None = None,
    enable_tts: bool = True,
    full_duplex: bool = True,
    input_device: int | str | None = None,
    output_device: int | str | None = None,
    input_channel: int = 0,
    output_channel: int | None = None,
    input_rate: int | None = None,
    output_rate: int | None = None,
    input_channels: int | None = None,
    output_channels: int | None = None,
    min_energy: float = 0.006,
    silence_ms: int = 300,
    max_segment_s: float = 4.0,
    language: str = "da",
    context: ContextProfile | None = None,
    sentence_level: bool = False,
    as_json: bool = False,
    stats: bool = False,
    broadcast: BroadcastConfig | None = None,
) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise DependencyMissingError(
            "audio input/output needs sounddevice and PortAudio.\n"
            "  uv:  uv sync --extra speech\n"
            "  apt: sudo apt install libportaudio2"
        ) from exc

    broadcast_session = None
    if broadcast is not None and broadcast.enabled:
        if not enable_tts:
            raise DanglishError("broadcast requires TTS; remove --no-tts")
        from ..broadcast import BroadcastSession

        broadcast_session = BroadcastSession(broadcast)
        broadcast_session.start()
        print(f"broadcast:    {broadcast_session.listener_url}", file=sys.stderr)
        print(f"operator QR:  {broadcast_session.qr_url}", file=sys.stderr)
        print(f"speaker page: {broadcast_session.speaker_url}", file=sys.stderr)
        if not broadcast.local:
            print(
                "broadcast:    hotspot mode; the Wi-Fi will be restored on exit",
                file=sys.stderr,
            )

    in_index = resolve_device(sd, input_device, "input")
    in_info = sd.query_devices(in_index, "input")
    capture_rate = choose_rate(in_info, input_rate)
    capture_channels = choose_input_channels(in_info, input_channels, input_channel)
    selected_input = channel_index(input_channel, capture_channels)
    print(f"audio input:  {describe_device(sd, in_index, 'input')}", file=sys.stderr)
    print(
        f"capture:      channel {selected_input} of {capture_channels} "
        f"at {capture_rate} Hz → {_SAMPLE_RATE} Hz VAD",
        file=sys.stderr,
    )

    output_stream = None
    played_rate: int | None = None
    played_channels: int | None = None
    if enable_tts:
        out_index = resolve_device(sd, output_device, "output")
        out_info = sd.query_devices(out_index, "output")
        played_rate = choose_rate(out_info, output_rate)
        played_channels = choose_output_channels(out_info, output_channels, output_channel)
        print(f"audio output: {describe_device(sd, out_index, 'output')}", file=sys.stderr)
        print(
            f"playback:     {played_channels} channel(s) at {played_rate} Hz",
            file=sys.stderr,
        )
        try:
            output_stream = sd.OutputStream(
                device=out_index,
                samplerate=played_rate,
                channels=played_channels,
                dtype="float32",
                blocksize=max(1, int(played_rate * 0.03)),
            )
            output_stream.start()
        except (sd.PortAudioError, ValueError) as exc:
            if output_stream is not None:
                with contextlib.suppress(Exception):
                    output_stream.close()
                output_stream = None
            raise DanglishError(
                f"could not open output device {out_index}: {exc}. "
                "Run `danglish audio-check` to test the device and channels."
            ) from exc

    try:
        actual_asr_model = asr_model or DEFAULT_ASR_MODEL
        print(
            f"loading Whisper {actual_asr_model!r} ({asr_compute_type}); "
            "the first run downloads the model ...",
            file=sys.stderr,
            flush=True,
        )
        asr = WhisperASR(
            actual_asr_model,
            compute_type=asr_compute_type,
            language=language,
            cpu_threads=config.model.intra_threads or 4,
        )

        tts = None
        if enable_tts:
            tts = create_tts(
                voice=tts_voice or config.tts.voice,
                backend=tts_backend or config.tts.backend,
            )
            print(
                f"loading English voice {tts.voice_name!r} ({tts.backend_name}) ...",
                file=sys.stderr,
                flush=True,
            )
            tts.warmup()

        speech = SpeechTranslator(
            asr,
            backend,
            tts,
            min_partial_chars=config.stream.min_partial_chars,
            max_pending_chars=config.stream.max_pending_chars,
            context=context if context is not None else get_profile(None),
            sentence_level=sentence_level,
        )
        if not 1.0 <= max_segment_s <= 30.0:
            raise DanglishError("max-segment-s must be between 1 and 30 seconds")
        vad = EnergyVAD(
            sample_rate=_SAMPLE_RATE,
            frame_ms=_FRAME_MS,
            end_silence_ms=silence_ms,
            max_utterance_s=max_segment_s,
            min_energy=min_energy,
        )

        muted = threading.Event()
        if os.environ.get("DANGLISH_CONTROL_STDIN") == "1":

            def control_loop() -> None:
                stream = sys.stdin
                if stream is None:
                    return
                for line in stream:
                    command = line.strip().lower()
                    if command == "mute":
                        muted.set()
                    elif command == "unmute":
                        muted.clear()
                    elif command == "toggle":
                        if muted.is_set():
                            muted.clear()
                        else:
                            muted.set()

            threading.Thread(target=control_loop, name="danglish-control", daemon=True).start()

        def play(audio: np.ndarray, sample_rate: int) -> None:
            if muted.is_set():
                return
            if broadcast_session is not None:
                try:
                    broadcast_chunk = max(1, int(sample_rate * 0.1))
                    for start in range(0, audio.size, broadcast_chunk):
                        if muted.is_set():
                            return
                        broadcast_session.write(audio[start : start + broadcast_chunk], sample_rate)
                except Exception as exc:
                    print(f"broadcast: {exc}", file=sys.stderr, flush=True)
            if output_stream is None or played_rate is None or played_channels is None:
                return
            resampled = resample_audio(audio, sample_rate, played_rate)
            if output_channel is None:
                data = np.repeat(resampled[:, None], played_channels, axis=1)
            else:
                selected = channel_index(output_channel, played_channels)
                data = np.zeros((resampled.size, played_channels), dtype=np.float32)
                data[:, selected] = resampled
            output_chunk = max(1, int(played_rate * 0.1))
            for start in range(0, data.shape[0], output_chunk):
                if muted.is_set():
                    return
                output_stream.write(data[start : start + output_chunk])

        def print_result(result: SpeechUtterance) -> None:
            if as_json:
                print(
                    json.dumps(
                        {
                            "type": "utterance",
                            "transcript": result.transcript,
                            "segments": [
                                {
                                    "source": segment.source,
                                    "target": segment.target,
                                    "tts_error": segment.tts_error,
                                }
                                for segment in result.segments
                            ],
                            "asr_ms": round(result.asr_seconds * 1000, 1),
                            "mt_ms": round(result.mt_seconds * 1000, 1),
                            "tts_ms": round(result.tts_seconds * 1000, 1),
                            "total_ms": round(result.total_seconds * 1000, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            else:
                print(f"da  {result.transcript}")
                for segment in result.segments:
                    print(f"en  {segment.target}")
                print(flush=True)

            if stats:
                print(
                    f"    asr {result.asr_seconds:.2f}s  "
                    f"mt {result.mt_seconds:.2f}s  "
                    f"tts {result.tts_seconds:.2f}s  "
                    f"total {result.total_seconds:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )

        def print_error(message: str) -> None:
            print(f"talk: {message}", file=sys.stderr, flush=True)

        def print_speech(duration: float) -> None:
            if as_json:
                print(
                    json.dumps({"type": "speech", "duration_s": round(duration, 2)}),
                    flush=True,
                )
            else:
                print(f"speech: {duration:.1f}s", file=sys.stderr, flush=True)

        runtime = TalkRuntime(
            speech,
            play,
            vad,
            full_duplex=full_duplex,
            on_utterance=print_result,
            on_speech=print_speech,
            on_error=print_error,
            sample_rate=_SAMPLE_RATE,
        )
        runtime.start()

        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if status:
                print(f"audio: {status}", file=sys.stderr)
            if indata.size == 0:
                return
            channel = min(selected_input, indata.shape[1] - 1)
            frame = indata[:, channel].copy()
            if capture_rate != _SAMPLE_RATE:
                frame = resample_audio(frame, capture_rate, _SAMPLE_RATE)
            runtime.capture(frame)

        print(
            f"ready — speech stack RSS {current_rss_mb():,.0f} MB",
            file=sys.stderr,
            flush=True,
        )
        print(
            f"chunking: {silence_ms} ms silence / {max_segment_s:.1f}s max segment",
            file=sys.stderr,
            flush=True,
        )
        if full_duplex:
            print(
                "full-duplex: listening continues while English audio plays. "
                "Use isolated input/output or an external echo canceller.",
                file=sys.stderr,
            )
        else:
            print(
                "half-duplex: the microphone is muted while English audio plays. "
                "Full-duplex is the default with isolated I/O.",
                file=sys.stderr,
            )
        print("listening — speak Danish, Ctrl-C to stop.", file=sys.stderr, flush=True)

        try:
            with sd.InputStream(
                samplerate=capture_rate,
                channels=capture_channels,
                dtype="float32",
                blocksize=max(1, int(capture_rate * 0.03)),
                device=in_index,
                callback=callback,
            ):
                runtime.run_vad_loop()
        except KeyboardInterrupt:
            pass
        except (sd.PortAudioError, ValueError) as exc:
            raise DanglishError(
                f"could not open input device {in_index}: {exc}. "
                "Run `danglish audio-check` to test the device and channels."
            ) from exc
        finally:
            runtime.stop()
            if output_stream is not None:
                with contextlib.suppress(Exception):
                    output_stream.abort()
            runtime.join()
            speech.close()
    finally:
        if output_stream is not None:
            with contextlib.suppress(Exception):
                output_stream.abort()
            with contextlib.suppress(Exception):
                output_stream.close()
        if broadcast_session is not None:
            with contextlib.suppress(Exception):
                broadcast_session.stop()
        print("\nstopped.", file=sys.stderr)


# Historical name; `talk` is the primary command.
run_mic = run_talk
