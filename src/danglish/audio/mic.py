"""Microphone-to-speaker loop: capture → VAD → ASR → MT → TTS → playback."""

from __future__ import annotations

import contextlib
import json
import sys
from typing import Any

import numpy as np

from ..backends.base import TranslatorBackend
from ..config import AppConfig
from ..errors import DanglishError, DependencyMissingError
from ..memory import current_rss_mb
from ..speech import SpeechTranslator, SpeechUtterance
from ..tts import DEFAULT_VOICE, PiperTTS
from .asr import WhisperASR
from .runtime import TalkRuntime
from .util import resample_audio
from .vad import EnergyVAD

_SAMPLE_RATE = 16_000
_FRAME_MS = 30
_BLOCK_SIZE = _SAMPLE_RATE * _FRAME_MS // 1000


def run_talk(
    backend: TranslatorBackend,
    config: AppConfig,
    *,
    asr_model: str = "small",
    asr_compute_type: str = "int8",
    tts_voice: str | None = None,
    enable_tts: bool = True,
    full_duplex: bool = False,
    input_device: int | None = None,
    output_device: int | None = None,
    min_energy: float = 0.006,
    silence_ms: int = 600,
    language: str = "da",
    as_json: bool = False,
    stats: bool = False,
) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise DependencyMissingError(
            "audio input/output needs sounddevice and PortAudio.\n"
            "  uv:  uv sync --extra speech\n"
            "  apt: sudo apt install libportaudio2"
        ) from exc

    print(
        f"loading Whisper {asr_model!r} ({asr_compute_type}); "
        "the first run downloads the model ...",
        file=sys.stderr,
        flush=True,
    )
    asr = WhisperASR(
        asr_model,
        compute_type=asr_compute_type,
        language=language,
        cpu_threads=config.model.intra_threads or 4,
    )

    tts = None
    if enable_tts:
        voice = tts_voice or config.tts.voice or DEFAULT_VOICE
        print(f"loading English voice {voice!r} ...", file=sys.stderr, flush=True)
        tts = PiperTTS(voice)
        tts.warmup()

    speech = SpeechTranslator(
        asr,
        backend,
        tts,
        min_partial_chars=config.stream.min_partial_chars,
        max_pending_chars=config.stream.max_pending_chars,
    )
    vad = EnergyVAD(
        sample_rate=_SAMPLE_RATE,
        frame_ms=_FRAME_MS,
        end_silence_ms=silence_ms,
        min_energy=min_energy,
    )

    def play(audio: np.ndarray, sample_rate: int) -> None:
        try:
            sd.play(
                audio,
                samplerate=sample_rate,
                device=output_device,
                blocking=True,
            )
        except sd.PortAudioError:
            info = (
                sd.query_devices(output_device, "output")
                if output_device is not None
                else sd.query_devices(kind="output")
            )
            target_rate = int(info["default_samplerate"])
            resampled = resample_audio(audio, sample_rate, target_rate)
            sd.play(
                resampled,
                samplerate=target_rate,
                device=output_device,
                blocking=True,
            )

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

    runtime = TalkRuntime(
        speech,
        play,
        vad,
        full_duplex=full_duplex,
        on_utterance=print_result,
        on_error=print_error,
        sample_rate=_SAMPLE_RATE,
    )
    runtime.start()

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(f"audio: {status}", file=sys.stderr)
        runtime.capture(indata[:, 0].copy())

    print(
        f"ready — speech stack RSS {current_rss_mb():,.0f} MB",
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
            "Use --full-duplex with isolated I/O to keep listening.",
            file=sys.stderr,
        )
    print("listening — speak Danish, Ctrl-C to stop.", file=sys.stderr, flush=True)

    try:
        with sd.InputStream(
            samplerate=_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=_BLOCK_SIZE,
            device=input_device,
            callback=callback,
        ):
            runtime.run_vad_loop()
    except KeyboardInterrupt:
        pass
    except (sd.PortAudioError, ValueError) as exc:
        raise DanglishError(
            f"could not open the audio device: {exc}. Try `danglish talk --list-devices`."
        ) from exc
    finally:
        runtime.stop()
        with contextlib.suppress(Exception):
            sd.stop()
        runtime.join()
        speech.close()
        print("\nstopped.", file=sys.stderr)


# Historical name; `talk` is the primary command.
run_mic = run_talk
