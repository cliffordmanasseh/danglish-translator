from __future__ import annotations

import os

import numpy as np
import pytest

from danglish.audio.util import read_wav, write_wav

pytestmark = pytest.mark.integration


def test_kokoro_produces_audible_speech() -> None:
    pytest.importorskip("kokoro_onnx")
    from pathlib import Path

    from danglish.tts import KokoroTTS

    model = Path("models/kokoro/kokoro-v1.0.int8.onnx")
    voices = Path("models/kokoro/voices-v1.0.bin")
    if not model.is_file() or not voices.is_file():
        pytest.skip("Kokoro model files are not present")

    tts = KokoroTTS("af_heart", model_path=model, voices_path=voices)
    try:
        audio = tts.synthesize("Good morning. This is a test of the natural voice.")
        assert audio.dtype == np.float32
        assert audio.size > tts.sample_rate * 0.8
        assert float(np.max(np.abs(audio))) > 0.01
        assert tts.sample_rate == 24_000
    finally:
        tts.close()


def test_piper_produces_audible_speech(tmp_path) -> None:
    pytest.importorskip("piper")
    from danglish.tts import PiperTTS

    voice = os.environ.get("DANGLISH_TTS_VOICE", "en_GB-alba-medium")
    tts = PiperTTS(voice)
    try:
        audio = tts.synthesize("Good morning. This is a test of the English voice.")
        assert audio.dtype == np.float32
        assert audio.size > tts.sample_rate * 1.0
        assert float(np.max(np.abs(audio))) > 0.01
        assert float(np.sqrt(np.mean(audio**2))) > 0.001

        path = write_wav(tmp_path / "out.wav", audio, tts.sample_rate)
        restored, rate = read_wav(path)
        assert rate == tts.sample_rate
        assert abs(restored.size - audio.size) <= 1
    finally:
        tts.close()
