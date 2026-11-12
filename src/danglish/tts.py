"""Local English speech synthesis with Piper.

The Piper runtime and voice files are optional. Voices are downloaded on first use
from the public ``rhasspy/piper-voices`` repository and cached by Hugging Face.
"""

from __future__ import annotations

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
