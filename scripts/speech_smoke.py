#!/usr/bin/env python3
"""Offline audio-to-audio smoke test: a WAV file in, an English WAV file out.

This exercises the same SpeechTranslator path as `danglish talk`, without needing a
microphone. Use a short clip: the whole file is treated as one utterance. Requires the
speech extra and a converted translation model.

Example:
    uv run --extra speech python scripts/speech_smoke.py danish.wav -o english.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from danglish.audio.asr import WhisperASR
from danglish.audio.util import read_wav, write_wav
from danglish.backends import create_backend
from danglish.config import load_config
from danglish.memory import current_rss_mb, peak_rss_mb
from danglish.speech import SpeechTranslator
from danglish.tts import DEFAULT_VOICE, PiperTTS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="mono 16-bit PCM WAV file with Danish speech")
    parser.add_argument("-o", "--output", type=Path, default=Path("english.wav"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--voice")
    args = parser.parse_args()

    audio, sample_rate = read_wav(args.input)
    config = load_config(args.config)
    if args.model_path is not None:
        config.model.path = args.model_path
    voice = args.voice or config.tts.voice or DEFAULT_VOICE

    print("loading models ...", file=sys.stderr, flush=True)
    backend = create_backend(config)
    asr = WhisperASR(args.asr_model, compute_type="int8", cpu_threads=4)
    tts = PiperTTS(voice)
    speech = SpeechTranslator(asr, backend, tts)

    started = time.perf_counter()
    try:
        result = speech.translate_utterance(audio, sample_rate=sample_rate)
        if result is None:
            print("no speech recognized", file=sys.stderr)
            return 1

        print(f"DA  {result.transcript}")
        pieces: list[np.ndarray] = []
        for segment in result.segments:
            print(f"EN  {segment.target}")
            if segment.audio is not None and segment.sample_rate is not None:
                pieces.append(segment.audio)
                pieces.append(np.zeros(int(0.12 * segment.sample_rate), dtype=np.float32))
            if segment.tts_error:
                print(f"tts error: {segment.tts_error}", file=sys.stderr)

        if pieces:
            combined = np.concatenate(pieces)
            write_wav(args.output, combined, tts.sample_rate)
            print(f"wrote {args.output} ({combined.size / tts.sample_rate:.2f}s)")

        wall = time.perf_counter() - started
        print(
            f"asr {result.asr_seconds:.2f}s  mt {result.mt_seconds:.2f}s  "
            f"tts {result.tts_seconds:.2f}s  wall {wall:.2f}s  "
            f"RSS {current_rss_mb():.0f} MB  peak {peak_rss_mb():.0f} MB",
            file=sys.stderr,
        )
        return 0
    finally:
        speech.close()
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
