from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from danglish.errors import DanglishError, DependencyMissingError
from danglish.tts import (
    DEFAULT_TINY_VOCA_VOICE,
    DEFAULT_VOICE,
    PiperTTS,
    TinyVocaTTS,
    _finished,
    _sentences,
    _shorter,
    create_tts,
    ensure_tiny_voca,
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


def test_ensure_tiny_voca_uses_bundled_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "onnx" / "model.onnx"
    model.parent.mkdir()
    model.write_bytes(b"onnx")
    phonemizer = tmp_path / "phonemizer"
    phonemizer.mkdir()
    (phonemizer / "lexicon.tsv").write_text("", encoding="utf-8")
    monkeypatch.setenv("DANGLISH_TINY_VOCA_DIR", str(tmp_path))

    assert ensure_tiny_voca() == (model, phonemizer)


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


def test_sentence_helpers() -> None:
    assert _sentences("One. Two!") == ["One.", "Two!"]
    assert _sentences("Dr. Smith is here. Hello!") == ["Dr. Smith is here.", "Hello!"]
    assert _finished("Hello") == "Hello."
    assert _finished("Hello.") == "Hello."
    assert _shorter("one, two") == ["one,", "two"]
    assert _shorter("one two three four") == ["one two", "three four"]
    assert _shorter("one") == []


def test_create_tts_defaults_to_tiny_voca(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeTinyVoca:
        def __init__(self, voice: str) -> None:
            calls.append(("tiny-voca", voice))

    monkeypatch.setattr("danglish.tts.TinyVocaTTS", FakeTinyVoca)

    tts = create_tts()

    assert isinstance(tts, FakeTinyVoca)
    assert calls == [("tiny-voca", DEFAULT_TINY_VOCA_VOICE)]


def test_create_tts_auto_selects_piper_for_piper_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakePiper:
        def __init__(self, voice: str) -> None:
            calls.append(voice)

    monkeypatch.setattr("danglish.tts.PiperTTS", FakePiper)

    tts = create_tts("en_GB-alba-medium")

    assert isinstance(tts, FakePiper)
    assert calls == ["en_GB-alba-medium"]


def test_create_tts_explicit_backend_uses_that_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class FakeTinyVoca:
        def __init__(self, voice: str) -> None:
            calls.append(("tiny-voca", voice))

    class FakePiper:
        def __init__(self, voice: str) -> None:
            calls.append(("piper", voice))

    monkeypatch.setattr("danglish.tts.TinyVocaTTS", FakeTinyVoca)
    monkeypatch.setattr("danglish.tts.PiperTTS", FakePiper)

    assert isinstance(create_tts("F2", "tiny-voca"), FakeTinyVoca)
    # A tiny-voca voice name with the Piper backend falls back to its default.
    assert isinstance(create_tts("M2", "piper"), FakePiper)
    assert calls == [("tiny-voca", "F2"), ("piper", DEFAULT_VOICE)]


def test_create_tts_rejects_unknown_backend() -> None:
    with pytest.raises(DanglishError, match="unknown TTS backend"):
        create_tts(backend="espeak")


def test_tiny_voca_synthesizes_and_limits_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phonemizer_dir = tmp_path / "phonemizer"
    phonemizer_dir.mkdir()
    (phonemizer_dir / "lexicon.tsv").write_text("", encoding="utf-8")
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")

    fake_dizione = types.ModuleType("dizione")

    class Phones:
        ids = (1, 2)
        stress = (0, 1)

        def __len__(self) -> int:
            return len(self.ids)

    fake_dizione.phonemize = lambda text: Phones()  # type: ignore[attr-defined]
    fake_dizione.reset = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dizione", fake_dizione)

    class FakeSession:
        def __init__(self, path: str, sess_options: object = None) -> None:
            self.path = path

        def run(self, output_names: object, feeds: dict[str, np.ndarray]) -> list[object]:
            assert feeds["tokens"].tolist() == [[1, 2]]
            assert feeds["voice"].tolist() == [1]  # M2
            return [np.array([[0.5, -0.5, 1.5]], dtype=np.float32), None]

    class FakeOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.SessionOptions = FakeOptions  # type: ignore[attr-defined]
    fake_ort.InferenceSession = FakeSession  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    tts = TinyVocaTTS("M2", model_path=model_path, phonemizer_dir=phonemizer_dir)
    audio = tts.synthesize("Hello.")

    assert audio.dtype == np.float32
    assert np.isclose(np.max(np.abs(audio)), 0.99)
    assert tts.synthesize("   ").size == 0
    assert tts.voice_name == "M2"
    assert tts.backend_name == "tiny-voca"
