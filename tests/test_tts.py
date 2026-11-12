from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from danglish.errors import DanglishError, DependencyMissingError
from danglish.tts import (
    DEFAULT_KOKORO_VOICE,
    DEFAULT_VOICE,
    KokoroTTS,
    PiperTTS,
    create_tts,
    ensure_kokoro_files,
    ensure_voice,
    parse_voice_name,
)


def test_parse_voice_name() -> None:
    spec = parse_voice_name("en_GB-alba-medium")

    assert spec.language == "en"
    assert spec.locale == "en_GB"
    assert spec.speaker_name == "alba"
    assert spec.quality == "medium"
    assert spec.repo_path == "en/en_GB/alba/medium/en_GB-alba-medium.onnx"
    assert DEFAULT_VOICE == "en_GB-alba-medium"


@pytest.mark.parametrize(
    "name",
    ["alba", "en_GB-alba", "en_GB-alba-ultra", "../etc/passwd", "en_GB-Alba-medium"],
)
def test_parse_invalid_voice_name(name: str) -> None:
    with pytest.raises(DanglishError, match="invalid Piper voice name"):
        parse_voice_name(name)


def test_ensure_voice_local_path(tmp_path: Path) -> None:
    model = tmp_path / "myvoice.onnx"
    config = tmp_path / "myvoice.onnx.json"
    model.write_bytes(b"onnx")
    config.write_text("{}", encoding="utf-8")

    assert ensure_voice(model) == (model, config)


def test_ensure_voice_missing_local_path(tmp_path: Path) -> None:
    with pytest.raises(DanglishError, match="not found"):
        ensure_voice(tmp_path / "missing.onnx")


def test_ensure_voice_downloads_expected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, filename: str, **kwargs: object) -> str:
        calls.append((repo_id, filename))
        destination = tmp_path / Path(filename).name
        destination.write_bytes(b"data")
        return str(destination)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    model, config = ensure_voice("en_GB-alba-medium")

    assert calls == [
        (
            "rhasspy/piper-voices",
            "en/en_GB/alba/medium/en_GB-alba-medium.onnx",
        ),
        (
            "rhasspy/piper-voices",
            "en/en_GB/alba/medium/en_GB-alba-medium.onnx.json",
        ),
    ]
    assert model.name == "en_GB-alba-medium.onnx"
    assert config.name == "en_GB-alba-medium.onnx.json"


def _install_fake_piper(monkeypatch: pytest.MonkeyPatch, chunks: list[object]) -> None:
    piper_module = types.ModuleType("piper")
    voice_module = types.ModuleType("piper.voice")

    class FakeVoice:
        config = SimpleNamespace(sample_rate=22_050)

        @classmethod
        def load(cls, model_path: str, config_path: str | None = None) -> FakeVoice:
            return cls()

        def synthesize(self, text: str, syn_config: object = None):
            yield from chunks

    voice_module.PiperVoice = FakeVoice  # type: ignore[attr-defined]
    piper_module.voice = voice_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", piper_module)
    monkeypatch.setitem(sys.modules, "piper.voice", voice_module)


def test_piper_tts_uses_float_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    config = tmp_path / "voice.onnx.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("danglish.tts.ensure_voice", lambda voice: (model, config))

    chunk = SimpleNamespace(
        sample_rate=22_050,
        audio_float_array=np.array([0.1, -0.2, 0.3], dtype=np.float32),
    )
    _install_fake_piper(monkeypatch, [chunk])

    tts = PiperTTS("en_GB-alba-medium")
    audio = tts.synthesize("Hello")

    assert tts.sample_rate == 22_050
    np.testing.assert_allclose(audio, [0.1, -0.2, 0.3])
    assert tts.synthesize("   ").size == 0


def test_piper_tts_falls_back_to_int16(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    config = tmp_path / "voice.onnx.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("danglish.tts.ensure_voice", lambda voice: (model, config))

    pcm = np.array([0, 16384, -32768], dtype=np.int16)
    chunk = SimpleNamespace(sample_rate=22_050, audio_int16_bytes=pcm.tobytes())
    _install_fake_piper(monkeypatch, [chunk])

    audio = PiperTTS().synthesize("Hello")

    np.testing.assert_allclose(audio, [0.0, 0.5, -1.0], atol=1e-4)


def test_piper_tts_empty_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    config = tmp_path / "voice.onnx.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("danglish.tts.ensure_voice", lambda voice: (model, config))
    _install_fake_piper(monkeypatch, [])

    assert PiperTTS().synthesize("Hello").size == 0


def test_missing_piper_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "piper", None)
    monkeypatch.setitem(sys.modules, "piper.voice", None)

    with pytest.raises(DependencyMissingError, match="piper-tts"):
        PiperTTS()


def test_ensure_kokoro_files_uses_local_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "kokoro-v1.0.int8.onnx").write_bytes(b"model")
    (tmp_path / "voices-v1.0.bin").write_bytes(b"voices")
    monkeypatch.setenv("DANGLISH_KOKORO_DIR", str(tmp_path))

    model, voices = ensure_kokoro_files()

    assert model == tmp_path / "kokoro-v1.0.int8.onnx"
    assert voices == tmp_path / "voices-v1.0.bin"


def test_create_tts_backend_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeKokoro:
        def __init__(self, voice: str, **kwargs: object) -> None:
            calls.append(("kokoro", voice))

    class FakePiper:
        def __init__(self, voice: str) -> None:
            calls.append(("piper", voice))

    monkeypatch.setattr("danglish.tts.KokoroTTS", FakeKokoro)
    monkeypatch.setattr("danglish.tts.PiperTTS", FakePiper)

    create_tts("kokoro", None)
    create_tts("piper", DEFAULT_KOKORO_VOICE)
    create_tts("kokoro", DEFAULT_VOICE)

    assert calls == [
        ("kokoro", DEFAULT_KOKORO_VOICE),
        ("piper", DEFAULT_VOICE),
        ("kokoro", DEFAULT_KOKORO_VOICE),
    ]


def test_kokoro_tts_uses_onnx_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model = tmp_path / "kokoro.onnx"
    voices = tmp_path / "voices.bin"
    model.write_bytes(b"model")
    voices.write_bytes(b"voices")
    monkeypatch.setattr("danglish.tts.ensure_kokoro_files", lambda *a: (model, voices))

    kokoro_module = types.ModuleType("kokoro_onnx")
    observed: dict[str, str] = {}

    class FakeKokoro:
        def __init__(self, model_path: str, voices_path: str) -> None:
            observed["model"] = model_path
            observed["voices"] = voices_path

        def create(self, text: str, **kwargs: object):
            observed["text"] = text
            observed["lang"] = str(kwargs["lang"])
            observed["voice"] = str(kwargs["voice"])
            return np.array([0.1, -0.2], dtype=np.float32), 24_000

    kokoro_module.Kokoro = FakeKokoro  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kokoro_onnx", kokoro_module)

    tts = KokoroTTS("bf_emma")
    audio = tts.synthesize("Hello")

    assert tts.sample_rate == 24_000
    assert observed["lang"] == "en-gb"
    assert observed["voice"] == "bf_emma"
    np.testing.assert_allclose(audio, [0.1, -0.2])


def test_missing_kokoro_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "kokoro_onnx", None)

    with pytest.raises(DependencyMissingError, match="kokoro-onnx"):
        KokoroTTS()
