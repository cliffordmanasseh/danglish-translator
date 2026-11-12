"""Local Danish speech recognition with faster-whisper."""

from __future__ import annotations

import numpy as np

from ..errors import DependencyMissingError
from .util import resample_audio

_TARGET_SAMPLE_RATE = 16_000

# Danish fine-tune of Whisper small, packaged as a CTranslate2 int8 model. It has
# the same memory footprint as stock small but is markedly better on Danish
# (model-card WER 0.078 vs 0.234 on a 21.5 s clip).
DEFAULT_ASR_MODEL = "lucasw89/whisper-small-danish-ct2-int8"


class WhisperASR:
    """Thin wrapper around faster-whisper for short, VAD-segmented utterances."""

    def __init__(
        self,
        model_size: str = DEFAULT_ASR_MODEL,
        *,
        compute_type: str = "int8",
        cpu_threads: int = 4,
        language: str = "da",
        beam_size: int = 1,
        vad_filter: bool = True,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise DependencyMissingError(
                "speech input needs faster-whisper.\n"
                "  uv:  uv sync --extra asr\n"
                '  pip: pip install "danglish[asr]"'
            ) from exc

        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
        )

    def transcribe(self, audio: np.ndarray, *, sample_rate: int = _TARGET_SAMPLE_RATE) -> str:
        samples = np.asarray(audio)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.dtype == np.int16:
            samples = samples.astype(np.float32) / 32768.0
        else:
            samples = samples.astype(np.float32, copy=False)

        if samples.size == 0:
            return ""
        if sample_rate != _TARGET_SAMPLE_RATE:
            samples = resample_audio(samples, sample_rate, _TARGET_SAMPLE_RATE)

        if self.vad_filter:
            # Use Silero as a speech-presence gate, then transcribe the original
            # audio. Letting the inner VAD trim the audio can cut the last
            # syllable and degrade the transcript; using it only as a gate keeps
            # the full utterance while rejecting noise/silence.
            try:
                from faster_whisper.vad import VadOptions, get_speech_timestamps

                speech = get_speech_timestamps(
                    samples,
                    VadOptions(),
                    sampling_rate=_TARGET_SAMPLE_RATE,
                )
                if not speech:
                    return ""
            except Exception:
                # VAD failure must not drop real speech; continue without the gate.
                pass

        segments, _info = self._model.transcribe(
            samples,
            language=self.language,
            task="transcribe",
            beam_size=self.beam_size,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
            temperature=0.0,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def close(self) -> None:
        self._model = None  # type: ignore[assignment]
