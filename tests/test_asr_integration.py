from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_whisper_loads_and_runs_on_silence() -> None:
    pytest.importorskip("faster_whisper")
    from danglish.audio.asr import WhisperASR

    asr = WhisperASR("tiny", compute_type="int8", cpu_threads=2)
    try:
        text = asr.transcribe(np.zeros(16_000, dtype=np.float32))
        assert isinstance(text, str)
    finally:
        asr.close()
