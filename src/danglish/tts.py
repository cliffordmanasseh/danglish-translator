"""Local English speech synthesis.

The default engine is tiny-voca, a 4.5M-parameter Apache-2.0 model that runs
through ONNX Runtime. It downloads a 21 MB ONNX graph and a 5 MB dizione
phonemizer on first use. Piper remains available as a legacy fallback.

Both engines return mono float32 audio and expose ``voice_name``,
``backend_name``, and ``sample_rate``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import DanglishError, DependencyMissingError

VOICE_REPO = "rhasspy/piper-voices"
DEFAULT_VOICE = "en_GB-alba-medium"
RECOMMENDED_VOICES = (
    "en_GB-alba-medium",
    "en_GB-cori-high",
)

TINY_VOCA_REPO = "n4ze3m/tiny-voca"
TINY_VOCA_ONNX = "onnx/model.onnx"
TINY_VOCA_VOICES = ("F2", "M2")
DEFAULT_TINY_VOCA_VOICE = "M2"
TINY_VOCA_SAMPLE_RATE = 24_000
TTS_BACKENDS = ("auto", "tiny-voca", "piper")

_PEAK = 0.99
_TINY_VOCA_MAX_TOKENS = 512

# Sentence splitting follows tiny-voca's infer.py (Apache-2.0).
_SPACE = re.compile(r"[\s\ufeff]+")
_SENTENCE_END = re.compile(r"""(?:(?<=[.!?…])|(?<=[.!?…]["')\]]))\s+(?=["'(\[]?[A-Z0-9])""")
_NOT_AN_END = re.compile(
    r"(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|no|inc|ltd|co|e\.g|i\.e|a\.m|p\.m)|\b[A-Z])\.$",
    re.IGNORECASE | re.ASCII,
)
_CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")
_TEXT_END = re.compile(r"[\s\ufeff]+$")
_CLOSING = "\"'”’)]»"

_VOICE_RE = re.compile(
    r"^(?P<locale>[a-z]{2}_[A-Z]{2})-(?P<name>[a-z0-9_]+)-(?P<quality>low|medium|high)$"
)


@dataclass(frozen=True, slots=True)
class VoiceSpec:
    name: str
    language: str
    locale: str
    speaker_name: str
    quality: str

    @property
    def repo_path(self) -> str:
        return f"{self.language}/{self.locale}/{self.speaker_name}/{self.quality}/{self.name}.onnx"


def parse_voice_name(name: str) -> VoiceSpec:
    match = _VOICE_RE.fullmatch(name)
    if match is None:
        raise DanglishError(
            f"invalid Piper voice name {name!r}. Expected a name like "
            "'en_GB-alba-medium', or a path to a local .onnx voice."
        )
    locale = match.group("locale")
    return VoiceSpec(
        name=name,
        language=locale.split("_")[0],
        locale=locale,
        speaker_name=match.group("name"),
        quality=match.group("quality"),
    )


def ensure_voice(voice: str | Path = DEFAULT_VOICE) -> tuple[Path, Path]:
    """Return local paths to the Piper ONNX model and its JSON config.

    Accepts either a voice name from the Piper repository or a local ``.onnx``
    path (with a sibling ``.onnx.json`` file).
    """

    candidate = Path(voice)
    if candidate.suffix == ".onnx":
        config = Path(str(candidate) + ".json")
        if not candidate.is_file():
            raise DanglishError(f"voice model not found: {candidate}")
        if not config.is_file():
            raise DanglishError(f"voice config not found: {config}")
        return candidate, config

    spec = parse_voice_name(str(voice))
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - core dependency
        raise DependencyMissingError(
            "huggingface-hub is required to download Piper voices"
        ) from exc

    try:
        model_path = hf_hub_download(repo_id=VOICE_REPO, filename=spec.repo_path)
        config_path = hf_hub_download(repo_id=VOICE_REPO, filename=spec.repo_path + ".json")
    except Exception as exc:
        raise DanglishError(
            f"could not download Piper voice {spec.name!r}: {exc}\n"
            "Check the name and your network connection, or pass a local .onnx path."
        ) from exc
    return Path(model_path), Path(config_path)


class PiperTTS:
    """Piper voice wrapper returning mono float32 audio."""

    backend_name = "piper"

    def __init__(self, voice: str | Path = DEFAULT_VOICE) -> None:
        try:
            from piper.voice import PiperVoice
        except ImportError as exc:
            raise DependencyMissingError(
                "English speech output needs piper-tts.\n"
                "  uv:  uv sync --extra speech\n"
                '  pip: pip install "danglish[speech]"'
            ) from exc

        self.voice_name = str(voice)
        model_path, config_path = ensure_voice(voice)
        try:
            self._voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        except Exception as exc:
            raise DanglishError(f"could not load Piper voice {self.voice_name!r}: {exc}") from exc
        self.sample_rate = int(getattr(self._voice.config, "sample_rate", 22_050))

    def __repr__(self) -> str:
        return f"PiperTTS(voice={self.voice_name!r}, sample_rate={self.sample_rate})"

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize English text into a mono float32 waveform in [-1, 1]."""

        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        pieces: list[np.ndarray] = []
        for chunk in self._voice.synthesize(text):
            audio = getattr(chunk, "audio_float_array", None)
            if audio is not None:
                pieces.append(np.asarray(audio, dtype=np.float32))
            else:
                raw = getattr(chunk, "audio_int16_bytes", None)
                if raw:
                    pieces.append(np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)

            chunk_rate = getattr(chunk, "sample_rate", None)
            if chunk_rate:
                self.sample_rate = int(chunk_rate)

        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces).astype(np.float32, copy=False)

    def warmup(self) -> None:
        """Synthesize a tiny phrase so the first real answer is fast."""

        self.synthesize("Ready.")

    def close(self) -> None:
        self._voice = None  # type: ignore[assignment]


def ensure_tiny_voca() -> tuple[Path, Path]:
    """Return local paths to the tiny-voca ONNX graph and dizione data.

    ``DANGLISH_TINY_VOCA_DIR`` can point at a bundled directory containing
    ``onnx/model.onnx`` and ``phonemizer/``. Otherwise the model is downloaded
    from Hugging Face on first use and cached.
    """

    override = os.environ.get("DANGLISH_TINY_VOCA_DIR")
    if override:
        directory = Path(override)
        model_path = directory / TINY_VOCA_ONNX
        phonemizer_dir = directory / "phonemizer"
        if not model_path.is_file():
            raise DanglishError(f"tiny-voca ONNX model not found at {model_path}")
        if not (phonemizer_dir / "lexicon.tsv").is_file():
            raise DanglishError(f"tiny-voca phonemizer data not found at {phonemizer_dir}")
        return model_path, phonemizer_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - core dependency
        raise DependencyMissingError(
            "huggingface-hub is required to download the tiny-voca model"
        ) from exc

    try:
        directory = Path(
            snapshot_download(
                repo_id=TINY_VOCA_REPO,
                allow_patterns=[TINY_VOCA_ONNX, "phonemizer/*"],
            )
        )
    except Exception as exc:
        raise DanglishError(
            f"could not download tiny-voca from {TINY_VOCA_REPO}: {exc}\n"
            "Check your network connection, or set DANGLISH_TINY_VOCA_DIR to a "
            "local bundle containing onnx/model.onnx and phonemizer/."
        ) from exc

    model_path = directory / TINY_VOCA_ONNX
    phonemizer_dir = directory / "phonemizer"
    if not model_path.is_file():
        raise DanglishError(f"tiny-voca ONNX model not found at {model_path}")
    if not (phonemizer_dir / "lexicon.tsv").is_file():
        raise DanglishError(f"tiny-voca phonemizer data not found at {phonemizer_dir}")
    return model_path, phonemizer_dir


def _sentences(text: str) -> list[str]:
    """Split text into sentences, keeping titles and initials intact."""

    parts: list[str] = []
    current = ""
    for piece in _SENTENCE_END.split(_SPACE.sub(" ", text).strip()):
        current = f"{current} {piece}" if current else piece
        if not _NOT_AN_END.search(current):
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return [part for part in parts if any(char.isalnum() for char in part)]


def _finished(text: str) -> str:
    """Add a full stop when the text ends without a mark.

    tiny-voca lengthens the last syllable and lets the pitch fall only when it
    sees terminal punctuation; without one the final word breaks off.
    """

    body = _TEXT_END.sub("", text)
    if not body:
        return text
    last = _SPACE.split(body)[-1]
    if any(char.isalnum() for char in last):
        marked = last.rstrip(_CLOSING)[-1:] in tuple(".!?…,;:")
    else:
        marked = any(char in ".!?…" for char in last) or last in (",", ";", ":")
    return text if marked else body + "."


def _shorter(text: str) -> list[str]:
    """Cut an overlong sentence after clauses, or halfway between words."""

    pieces = _CLAUSE_END.split(text)
    if len(pieces) > 1:
        return pieces
    words = text.split(" ")
    if len(words) < 2:
        return []
    half = (len(words) + 1) // 2
    return [" ".join(words[:half]), " ".join(words[half:])]


class TinyVocaTTS:
    """tiny-voca ONNX synthesis returning mono float32 audio at 24 kHz."""

    backend_name = "tiny-voca"
    sample_rate = TINY_VOCA_SAMPLE_RATE

    def __init__(
        self,
        voice: str = DEFAULT_TINY_VOCA_VOICE,
        *,
        model_path: str | Path | None = None,
        phonemizer_dir: str | Path | None = None,
        threads: int = 2,
    ) -> None:
        if voice not in TINY_VOCA_VOICES:
            raise DanglishError(
                f"tiny-voca voice must be one of {', '.join(TINY_VOCA_VOICES)}, not {voice!r}"
            )
        self.voice_name = voice
        self._voice_id = TINY_VOCA_VOICES.index(voice)

        if model_path is None or phonemizer_dir is None:
            default_model, default_phonemizer = ensure_tiny_voca()
            model_path = model_path or default_model
            phonemizer_dir = phonemizer_dir or default_phonemizer

        self._model_path = Path(model_path)
        self._phonemizer_dir = Path(phonemizer_dir)
        if not self._phonemizer_dir.joinpath("lexicon.tsv").is_file():
            raise DanglishError(f"tiny-voca phonemizer data not found at {self._phonemizer_dir}")
        if not self._model_path.is_file():
            raise DanglishError(f"tiny-voca ONNX model not found at {self._model_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DependencyMissingError(
                "English speech output needs onnxruntime.\n"
                "  uv:  uv sync --extra speech\n"
                '  pip: pip install "danglish[speech]"'
            ) from exc

        try:
            import dizione
        except ImportError as exc:
            raise DependencyMissingError(
                "English speech output needs dizione (the tiny-voca phonemizer).\n"
                "  uv:  uv sync --extra speech\n"
                '  pip: pip install "danglish[speech]"'
            ) from exc

        # dizione reads its built data directory from this variable.
        os.environ["DIZIONE_DATA"] = str(self._phonemizer_dir)
        dizione.reset()
        self._dizione = dizione

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        try:
            self._session = ort.InferenceSession(str(self._model_path), sess_options=options)
        except Exception as exc:
            raise DanglishError(
                f"could not load tiny-voca model {self._model_path}: {exc}"
            ) from exc
        self._max_tokens = _TINY_VOCA_MAX_TOKENS

    def __repr__(self) -> str:
        return f"TinyVocaTTS(voice={self.voice_name!r}, sample_rate={self.sample_rate})"

    def _speak(self, phones: object) -> np.ndarray:
        if self._session is None:
            raise DanglishError("tiny-voca model is closed")
        ids = np.array([phones.ids], dtype=np.int64)  # type: ignore[attr-defined]
        stress = np.array([phones.stress], dtype=np.int64)  # type: ignore[attr-defined]

        def scalar(value: float) -> np.ndarray:
            return np.array([value], dtype=np.float32)

        feeds = {
            "tokens": ids,
            "stress": stress,
            "voice": np.array([self._voice_id], dtype=np.int64),
            "speed": scalar(1.0),
            "pitch_shift": scalar(0.0),
            "pitch_range": scalar(1.0),
        }
        try:
            audio, _ = self._session.run(None, feeds)
        except Exception as exc:
            raise DanglishError(f"tiny-voca synthesis failed: {exc}") from exc

        wave = np.asarray(audio[0], dtype=np.float32)
        if wave.size:
            peak = float(np.abs(wave).max())
            if peak > _PEAK:
                wave = wave * (_PEAK / peak)
        return wave

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize English text into a mono float32 waveform in [-1, 1]."""

        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        waves: list[np.ndarray] = []
        limit = self._max_tokens - 12

        def read(piece: str) -> None:
            try:
                phones = self._dizione.phonemize(piece)
            except ValueError:
                return
            if len(phones) <= limit:
                waves.append(self._speak(phones))
                return
            pieces = _shorter(piece)
            if not pieces:
                raise DanglishError(f"one word is too long to speak: {piece[:40]!r}")
            for part in pieces:
                read(part)

        for sentence in _sentences(_finished(text)):
            read(sentence)
        if not waves:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(waves).astype(np.float32, copy=False)

    def warmup(self) -> None:
        """Synthesize a tiny phrase so the first real answer is fast."""

        self.synthesize("Ready.")

    def close(self) -> None:
        self._session = None  # type: ignore[assignment]


def create_tts(
    voice: str | None = None,
    backend: str | None = None,
) -> PiperTTS | TinyVocaTTS:
    """Build the selected TTS engine.

    ``backend="auto"`` picks tiny-voca unless the voice is a Piper voice name or
    a local Piper ``.onnx`` path. This keeps existing configs working while
    making tiny-voca the default for new installs.
    """

    selected = (backend or "auto").strip().lower()
    if selected not in TTS_BACKENDS:
        raise DanglishError(
            f"unknown TTS backend {backend!r}; choose one of {', '.join(TTS_BACKENDS)}"
        )

    if selected == "auto":
        selected = "tiny-voca" if not voice or voice in TINY_VOCA_VOICES else "piper"

    if selected == "tiny-voca":
        if voice and voice not in TINY_VOCA_VOICES:
            raise DanglishError(
                f"tiny-voca voices are {', '.join(TINY_VOCA_VOICES)}, not {voice!r}"
            )
        return TinyVocaTTS(voice or DEFAULT_TINY_VOCA_VOICE)

    if not voice or voice in TINY_VOCA_VOICES:
        voice = DEFAULT_VOICE
    return PiperTTS(voice)
