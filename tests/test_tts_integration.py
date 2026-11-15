from __future__ import annotations

import os

import numpy as np
import pytest

from danglish.audio.util import read_wav, write_wav

pytestmark = pytest.mark.integration


def test_tiny_voca_produces_audible_speech(tmp_path) -> None:
    from danglish.tts import TinyVocaTTS

    voice = os.environ.get("DANGLISH_TTS_VOICE", "M2")
    tts = TinyVocaTTS(voice)
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


def test_piper_produces_audible_speech(tmp_path) -> None:
    pytest.importorskip("piper")
    from danglish.tts import PiperTTS

    voice = os.environ.get("DANGLISH_PIPER_VOICE", "en_GB-alba-medium")
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
