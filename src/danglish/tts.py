"""Local English speech synthesis.

Kokoro-82M is the default because it sounds substantially more natural than
Piper while still running on CPU. Piper remains available as a faster, smaller
fallback. Both are optional dependencies.

Voice/model files are downloaded on first use and cached locally, or supplied by
the desktop bundle via explicit paths.
"""

from __future__ import annotations

import os
import re
import urllib.request
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

DEFAULT_TTS_BACKEND = "kokoro"
DEFAULT_KOKORO_VOICE = "af_heart"
KOKORO_VOICES = (
    "af_heart",
    "af_bella",
    "bf_emma",
    "bm_george",
)
KOKORO_MODEL_NAME = "kokoro-v1.0.int8.onnx"
KOKORO_VOICES_NAME = "voices-v1.0.bin"
KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/kokoro-v1.0.int8.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin"
)

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


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"downloading {destination.name} ...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            last_percent = -10
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total:
                    percent = int(done * 100 / total)
                    if percent >= last_percent + 10:
                        print(f"  {percent}%", flush=True)
                        last_percent = percent
        partial.replace(destination)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise DanglishError(f"could not download {destination.name}: {exc}") from exc


def ensure_kokoro_files(
    model_path: str | Path | None = None,
    voices_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Return local paths to the Kokoro ONNX model and voices bundle."""

    model = Path(model_path) if model_path else None
    voices = Path(voices_path) if voices_path else None

    if model is None or voices is None:
        local = Path(os.environ.get("DANGLISH_KOKORO_DIR", "models/kokoro"))
        cache = (
            Path(
                os.environ.get(
                    "DANGLISH_CACHE",
                    str(Path.home() / ".cache" / "danglish"),
                )
            )
            / "kokoro"
        )
        base = local if local.is_dir() else cache
        if model is None:
            model = base / KOKORO_MODEL_NAME
        if voices is None:
            voices = base / KOKORO_VOICES_NAME

    if not model.is_file():
        _download(KOKORO_MODEL_URL, model)
    if not voices.is_file():
        _download(KOKORO_VOICES_URL, voices)
    return model, voices


class KokoroTTS:
    """Kokoro-82M ONNX voice wrapper returning mono float32 audio."""

    def __init__(
        self,
        voice: str = DEFAULT_KOKORO_VOICE,
        *,
        model_path: str | Path | None = None,
        voices_path: str | Path | None = None,
        speed: float = 1.0,
    ) -> None:
        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise DependencyMissingError(
                "the natural English voice needs kokoro-onnx.\n"
                "  uv:  uv sync --extra speech\n"
                '  pip: pip install "danglish[speech]"'
            ) from exc

        self.voice_name = voice
        self.speed = speed
        self.sample_rate = 24_000
        self.lang = "en-gb" if voice.startswith("b") else "en-us"
        model, voices = ensure_kokoro_files(model_path, voices_path)
        try:
            self._kokoro = Kokoro(str(model), str(voices))
        except Exception as exc:
            raise DanglishError(f"could not load Kokoro voice model: {exc}") from exc

    def __repr__(self) -> str:
        return f"KokoroTTS(voice={self.voice_name!r}, sample_rate={self.sample_rate})"

    def synthesize(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        audio, sample_rate = self._kokoro.create(
            text,
            voice=self.voice_name,
            speed=self.speed,
            lang=self.lang,
        )
        self.sample_rate = int(sample_rate)
        return np.asarray(audio, dtype=np.float32)

    def warmup(self) -> None:
        self.synthesize("Ready.")

    def close(self) -> None:
        self._kokoro = None  # type: ignore[assignment]


class PiperTTS:
    """Piper voice wrapper returning mono float32 audio."""

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


def create_tts(
    backend: str = DEFAULT_TTS_BACKEND,
    voice: str | None = None,
    *,
    model_path: str | Path | None = None,
    voices_path: str | Path | None = None,
    speed: float = 1.0,
):
    """Create a TTS backend by name.

    Voice names are backend-specific. If a default voice from the other backend is
    passed (common when switching ``--tts-backend``), substitute the appropriate
    default rather than failing.
    """

    backend = (backend or DEFAULT_TTS_BACKEND).lower()
    if backend == "kokoro":
        if not voice or voice == DEFAULT_VOICE or _VOICE_RE.fullmatch(str(voice)):
            voice = DEFAULT_KOKORO_VOICE
        return KokoroTTS(
            voice,
            model_path=model_path,
            voices_path=voices_path,
            speed=speed,
        )
    if backend == "piper":
        if not voice or voice in KOKORO_VOICES:
            voice = DEFAULT_VOICE
        return PiperTTS(voice)
    raise DanglishError(f"unknown TTS backend {backend!r}; choose 'kokoro' or 'piper'")
