"""Audio-device selection and channel helpers.

PortAudio's ALSA default is not necessarily the same as the desktop's default
device. On Linux/PipeWire we prefer the PulseAudio host API's default, which
follows the desktop audio settings (and therefore a plugged-in UAD interface).
"""

from __future__ import annotations

import re
from typing import Any, Literal

DeviceKind = Literal["input", "output"]


def _normalize_name(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def resolve_device(sd: Any, requested: int | str | None, kind: DeviceKind) -> int:
    """Return the device index to use, preferring the desktop default.

    ``requested`` may be an index, a numeric string, or a case-insensitive name
    substring such as ``"Volt 4"`` or ``"Default Source"``. Names are stable across
    processes, unlike PortAudio indices when PipeWire nodes come and go.
    """

    if isinstance(requested, str):
        text = requested.strip()
        if not text:
            raise ValueError("device name must not be empty")
        if text.isdigit():
            requested = int(text)
        else:
            return _resolve_by_name(sd, text, kind)

    if requested is not None:
        info = sd.query_devices(requested, kind)
        channels = info["max_input_channels" if kind == "input" else "max_output_channels"]
        if channels < 1:
            raise ValueError(f"device {requested} has no {kind} channels")
        return int(requested)

    for hostapi in sd.query_hostapis():
        if hostapi["name"].lower().startswith("pulseaudio"):
            key = "default_input_device" if kind == "input" else "default_output_device"
            index = int(hostapi.get(key, -1))
            if index >= 0:
                return index

    info = sd.query_devices(kind=kind)
    return int(info["index"])


def _resolve_by_name(sd: Any, text: str, kind: DeviceKind) -> int:
    needle = _normalize_name(text)
    candidates: list[int] = []
    for index, info in enumerate(sd.query_devices()):
        name = str(info["name"])
        lowered = name.lower()
        if kind == "input":
            if info["max_input_channels"] < 1 or "monitor" in lowered:
                continue
        elif info["max_output_channels"] < 1:
            continue
        if needle in _normalize_name(name):
            candidates.append(index)

    if not candidates:
        raise ValueError(f"no {kind} device matching {text!r}; run `danglish audio-check --list`")

    pulse_index = None
    for index, hostapi in enumerate(sd.query_hostapis()):
        if hostapi["name"].lower().startswith("pulseaudio"):
            pulse_index = index
            break
    if pulse_index is not None:
        pulse_candidates = [
            index for index in candidates if sd.query_devices(index)["hostapi"] == pulse_index
        ]
        if pulse_candidates:
            candidates = pulse_candidates

    if len(candidates) > 1:
        prefix = "input" if kind == "input" else "output"
        preferred = [
            index for index in candidates if prefix in str(sd.query_devices(index)["name"]).lower()
        ]
        if len(preferred) == 1:
            candidates = preferred

    if len(candidates) > 1:
        choices = "; ".join(f"{index}: {sd.query_devices(index)['name']}" for index in candidates)
        raise ValueError(f"ambiguous {kind} device {text!r}: {choices}")

    return candidates[0]


def describe_device(sd: Any, index: int | None, kind: DeviceKind) -> str:
    if index is None:
        return "none"
    info = sd.query_devices(index, kind)
    channels = info["max_input_channels" if kind == "input" else "max_output_channels"]
    rate = int(info["default_samplerate"])
    hostapi = sd.query_hostapis(info["hostapi"])["name"]
    return f"{index} — {info['name']} ({channels} ch, {rate} Hz, {hostapi})"


def choose_rate(info: dict[str, Any], requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    return int(info["default_samplerate"])


def choose_input_channels(info: dict[str, Any], requested: int | None, channel: int) -> int:
    maximum = int(info["max_input_channels"])
    if requested is not None:
        return max(1, min(maximum, requested))
    # Open at least two channels when available: some interfaces route the first
    # microphone through a stereo pair, and this also works for mono devices.
    return max(1, min(maximum, max(2, channel + 1)))


def choose_output_channels(info: dict[str, Any], requested: int | None, channel: int | None) -> int:
    maximum = int(info["max_output_channels"])
    if requested is not None:
        return max(1, min(maximum, requested))
    if maximum <= 1:
        return 1
    if channel is None:
        return 2
    return max(2, min(maximum, channel + 1))


def channel_index(requested: int | None, available: int) -> int:
    if available <= 0:
        return 0
    if requested is None:
        return 0
    return max(0, min(available - 1, requested))
