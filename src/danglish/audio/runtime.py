"""Concurrent runtime for the microphone → ASR → MT → TTS → speaker loop.

Three concurrent stages keep capture alive while the previous turn is being
translated or spoken:

* the PortAudio callback pushes raw frames into a bounded queue;
* the VAD loop segments utterances and submits them to a translation worker;
* the worker runs ASR → MT → TTS and queues audio;
* a playback thread plays queued audio in order.

In the default half-duplex mode the capture gate drops frames while English audio
is playing. With ``full_duplex=True`` capture never pauses; this is intended for
isolated input/output hardware or an external echo canceller.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
import time
from collections.abc import Callable

import numpy as np

from ..speech import SpeechSegment, SpeechTranslator, SpeechUtterance
from .vad import EnergyVAD

PlaybackFn = Callable[[np.ndarray, int], None]
UtteranceFn = Callable[[SpeechUtterance], None]
SpeechFn = Callable[[float], None]
ErrorFn = Callable[[str], None]


class PlaybackGate:
    """Tracks queued audio and decides whether capture should continue.

    The gate counts *all* audio segments registered for playback. For a turn with
    several sentences, the gate stays closed until the last segment has finished,
    avoiding a brief unmute between sentences.
    """

    def __init__(self, full_duplex: bool) -> None:
        self.full_duplex = full_duplex
        self.active = threading.Event()
        self.reset_requested = threading.Event()
        self._pending = 0
        self._lock = threading.Lock()

    def should_capture(self) -> bool:
        return self.full_duplex or not self.active.is_set()

    def register(self, count: int) -> None:
        if count <= 0:
            return
        with self._lock:
            self._pending += count
            self.active.set()

    def finished_one(self) -> None:
        with self._lock:
            if self._pending > 0:
                self._pending -= 1
            if self._pending == 0:
                self.active.clear()
                if not self.full_duplex:
                    self.reset_requested.set()

    def consume_reset(self) -> bool:
        if self.reset_requested.is_set():
            self.reset_requested.clear()
            return True
        return False

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending


class TalkRuntime:
    """Threaded runtime independent of PortAudio and concrete models.

    Feed frames with :meth:`capture` from the audio callback and run
    :meth:`run_vad_loop` on the main thread. Call :meth:`start` first and
    :meth:`stop` / :meth:`join` on shutdown.
    """

    def __init__(
        self,
        speech: SpeechTranslator,
        play: PlaybackFn,
        vad: EnergyVAD,
        *,
        full_duplex: bool = False,
        on_utterance: UtteranceFn | None = None,
        on_speech: SpeechFn | None = None,
        on_error: ErrorFn | None = None,
        sample_rate: int = 16_000,
        frame_queue_size: int = 512,
        utterance_queue_size: int = 16,
        playback_queue_size: int = 64,
    ) -> None:
        self.gate = PlaybackGate(full_duplex)
        self.full_duplex = full_duplex
        self._speech = speech
        self._play = play
        self._vad = vad
        self._on_utterance = on_utterance or _default_print
        self._on_speech = on_speech
        self._on_error = on_error or _default_error
        self._sample_rate = sample_rate

        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=frame_queue_size)
        self._utterances: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=utterance_queue_size)
        self._playback: queue.Queue[SpeechSegment | None] = queue.Queue(maxsize=playback_queue_size)
        self._stop = threading.Event()
        self._dropped_frames = 0
        self._backlog_warned = False
        self._worker: threading.Thread | None = None
        self._player: threading.Thread | None = None

    @property
    def dropped_frames(self) -> int:
        return self._dropped_frames

    def start(self) -> None:
        if self._worker is not None:
            raise RuntimeError("TalkRuntime is already started")
        self._worker = threading.Thread(
            target=self._translate_loop, name="danglish-translate", daemon=True
        )
        self._player = threading.Thread(
            target=self._playback_loop, name="danglish-playback", daemon=True
        )
        self._worker.start()
        self._player.start()

    def capture(self, frame: np.ndarray) -> None:
        """Called from the audio callback; must never block."""

        if not self.gate.should_capture():
            self._dropped_frames += 1
            return
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            self._dropped_frames += 1

    def run_vad_loop(self) -> None:
        """Consume raw frames until stopped. Meant to run on the main thread."""

        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                self._apply_pending_reset()
                continue

            if frame is None:
                continue

            if self.gate.consume_reset():
                self._vad.reset()
                self._drain_frames()
                # The current frame may have been captured just before the gate
                # closed; start clean with the next one.
                continue

            for utterance in self._vad.push(frame):
                if self._on_speech is not None:
                    self._on_speech(len(utterance) / self._sample_rate)
                self._submit_utterance(utterance)

    def stop(self) -> None:
        self._stop.set()
        for target in (self._frames, self._utterances, self._playback):
            with contextlib.suppress(queue.Full):
                target.put_nowait(None)

    def join(self, timeout: float = 3.0) -> None:
        for thread in (self._worker, self._player):
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout)

    def _apply_pending_reset(self) -> None:
        if self.gate.consume_reset():
            self._vad.reset()
            self._drain_frames()

    def _drain_frames(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _submit_utterance(self, utterance: np.ndarray) -> None:
        while not self._stop.is_set():
            try:
                self._utterances.put(utterance, timeout=0.2)
                return
            except queue.Full:
                if not self._backlog_warned:
                    self._on_error(
                        "translation is backlogged; capture may be delayed. "
                        "Consider a smaller ASR model or fewer threads."
                    )
                    self._backlog_warned = True

    def _translate_loop(self) -> None:
        while not self._stop.is_set():
            try:
                utterance = self._utterances.get(timeout=0.2)
            except queue.Empty:
                continue
            if utterance is None:
                break

            try:
                result = self._speech.translate_utterance(utterance, sample_rate=self._sample_rate)
            except Exception as exc:  # keep the session alive after one bad turn
                self._on_error(f"translation failed: {type(exc).__name__}: {exc}")
                continue

            if result is None:
                continue

            for segment in result.segments:
                if segment.tts_error:
                    self._on_error(f"tts: {segment.tts_error}")

            try:
                self._on_utterance(result)
            except Exception as exc:
                self._on_error(f"output error: {type(exc).__name__}: {exc}")

            playable = [
                segment
                for segment in result.segments
                if segment.audio is not None
                and segment.audio.size > 0
                and segment.sample_rate is not None
                and segment.tts_error is None
            ]
            self.gate.register(len(playable))

            for segment in playable:
                if self._stop.is_set():
                    break
                while not self._stop.is_set():
                    try:
                        self._playback.put(segment, timeout=0.2)
                        break
                    except queue.Full:
                        continue

    def _playback_loop(self) -> None:
        while not self._stop.is_set():
            try:
                segment = self._playback.get(timeout=0.2)
            except queue.Empty:
                continue
            if segment is None:
                break

            try:
                if (
                    segment.audio is not None
                    and segment.audio.size > 0
                    and segment.sample_rate is not None
                ):
                    self._play(segment.audio, segment.sample_rate)
            except Exception as exc:
                self._on_error(f"playback failed: {type(exc).__name__}: {exc}")
            finally:
                # In half-duplex, let the room decay before reopening the mic.
                # This is the last queued segment when pending == 1.
                if not self.gate.full_duplex and self.gate.pending == 1:
                    time.sleep(0.15)
                self.gate.finished_one()


def _default_print(result: SpeechUtterance) -> None:
    print(f"da  {result.transcript}")
    for segment in result.segments:
        print(f"en  {segment.target}")
    print(flush=True)


def _default_error(message: str) -> None:
    print(f"runtime: {message}", file=sys.stderr, flush=True)
