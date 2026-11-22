"""A small, dependency-free energy VAD for microphone input.

This is not a speech recognizer. It only decides where an utterance starts and ends
so that faster-whisper does not have to process long stretches of silence. It is
intentionally simple and cheap: RMS energy with an adaptive noise floor.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class EnergyVAD:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        frame_ms: int = 30,
        start_frames: int = 3,
        end_silence_ms: int = 600,
        max_utterance_s: float = 15.0,
        min_utterance_s: float = 0.3,
        min_energy: float = 0.003,
        energy_multiplier: float = 3.0,
    ) -> None:
        if sample_rate <= 0 or frame_ms <= 0:
            raise ValueError("sample_rate and frame_ms must be positive")
        if start_frames < 1:
            raise ValueError("start_frames must be at least 1")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.start_frames = start_frames
        self.end_silence_frames = max(1, round(end_silence_ms / frame_ms))
        self.max_frames = max(1, round(max_utterance_s * 1000 / frame_ms))
        self.min_samples = int(min_utterance_s * sample_rate)
        self.min_energy = min_energy
        self.energy_multiplier = energy_multiplier

        self.noise_floor = max(1e-4, min_energy / 4.0)
        self.active = False
        self._ring: deque[np.ndarray] = deque(maxlen=max(1, round(500 / frame_ms)))
        self._buffer: list[np.ndarray] = []
        self._rms: list[float] = []
        self._speech_run = 0
        self._silence_frames = 0

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        """Feed one audio frame; return zero or more completed utterances."""

        samples = np.asarray(frame, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.size == 0:
            return []
        # Some microphones carry a constant DC offset that would otherwise count
        # as loud, permanent "speech".
        samples = samples - float(samples.mean())

        rms = float(np.sqrt(np.mean(np.square(samples))) + 1e-12)
        threshold = max(self.min_energy, self.noise_floor * self.energy_multiplier)
        is_speech = rms >= threshold
        self._ring.append(samples.copy())

        completed: list[np.ndarray] = []

        if not self.active:
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self.start_frames:
                    self.active = True
                    self._buffer = list(self._ring)
                    self._silence_frames = 0
            else:
                self._speech_run = 0
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
            return completed

        self._buffer.append(samples.copy())
        self._rms.append(rms)
        if is_speech:
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        if self._silence_frames >= self.end_silence_frames or len(self._buffer) >= self.max_frames:
            if self._silence_frames < self.end_silence_frames:
                self._learn_steady_noise(threshold)
            utterance = self._finish()
            if utterance is not None:
                completed.append(utterance)

        return completed

    def flush(self) -> list[np.ndarray]:
        """Finish an in-progress utterance, if any."""

        if not self.active:
            return []
        utterance = self._finish()
        return [utterance] if utterance is not None else []

    def reset(self) -> None:
        """Drop any in-progress audio without emitting it."""

        self.active = False
        self._buffer = []
        self._ring.clear()
        self._speech_run = 0
        self._silence_frames = 0

    def _learn_steady_noise(self, threshold: float) -> None:
        """A forced cut with no pauses at all means steady noise, not speech.

        Real speech has quiet gaps between words, so its low-percentile energy
        stays below the threshold. If even the quiet frames are above it, treat
        that level as the new noise floor.
        """

        if len(self._rms) < 10:
            return
        quiet = float(np.percentile(self._rms, 20))
        if quiet >= threshold:
            self.noise_floor = max(self.noise_floor, quiet)

    def _finish(self) -> np.ndarray | None:
        self._rms = []
        buffer = self._buffer

        # Keep a short tail of silence: ASR models often need a little context
        # after the last phoneme.
        keep_silence = max(1, round(150 / self.frame_ms))
        trim = max(0, self._silence_frames - keep_silence)
        if trim:
            buffer = buffer[: len(buffer) - trim]

        audio = (
            np.concatenate(buffer).astype(np.float32, copy=False)
            if buffer
            else np.zeros(0, dtype=np.float32)
        )

        self.active = False
        self._buffer = []
        self._speech_run = 0
        self._silence_frames = 0

        if audio.size < self.min_samples:
            return None
        return audio
