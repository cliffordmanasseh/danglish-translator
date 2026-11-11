"""Interactive audio-device diagnostics for UAD/interface troubleshooting."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .device import (
    channel_index,
    choose_input_channels,
    choose_output_channels,
    choose_rate,
    describe_device,
    resolve_device,
)


def _level(rms: float) -> str:
    if rms >= 0.01:
        return "good"
    if rms >= 0.002:
        return "quiet"
    return "silent"


def list_devices(
    sd: Any, *, input_index: int | None = None, output_index: int | None = None
) -> None:
    print("Audio devices (0-based indices):")
    print(f"{'':2} {'idx':>3}  {'name':<46} {'in':>3} {'out':>3}  {'rate':>6}  host API")
    for index, info in enumerate(sd.query_devices()):
        marker = ""
        if index == input_index:
            marker += "IN"
        if index == output_index:
            marker += "OUT"
        name = str(info["name"])
        if len(name) > 46:
            name = name[:43] + "..."
        print(
            f"{marker:>2} {index:>3}  {name:<46} "
            f"{info['max_input_channels']:>3} {info['max_output_channels']:>3}  "
            f"{int(info['default_samplerate']):>6}  {sd.query_hostapis(info['hostapi'])['name']}"
        )
    print()


def check_input(
    sd: Any,
    device: int | str | None = None,
    *,
    seconds: float = 3.0,
    channels: int | None = None,
    channel: int = 0,
    sample_rate: int | None = None,
) -> int:
    """Record *seconds* from the input and print per-channel levels.

    Returns the channel with the highest RMS, or -1 if the device could not be
    opened.
    """

    index = resolve_device(sd, device, "input")
    info = sd.query_devices(index, "input")
    rate = choose_rate(info, sample_rate)
    open_channels = choose_input_channels(info, channels, channel)
    print(f"Input:  {describe_device(sd, index, 'input')}")
    print(f"Opening {open_channels} channel(s) at {rate} Hz for {seconds:.1f}s.")
    print("Speak or play audio into the interface now ...", flush=True)
    time.sleep(0.4)

    frames: list[np.ndarray] = []
    got = 0
    overflows = 0
    block = max(1, int(rate * 0.03))
    try:
        with sd.InputStream(
            device=index,
            samplerate=rate,
            channels=open_channels,
            dtype="float32",
            blocksize=block,
        ) as stream:
            while got < int(rate * seconds):
                data, overflow = stream.read(block)
                frames.append(data.copy())
                got += len(data)
                if overflow:
                    overflows += 1
    except Exception as exc:
        print(f"could not open/read input device {index}: {type(exc).__name__}: {exc}")
        return -1

    audio = np.concatenate(frames)
    rms_values = np.sqrt(np.mean(audio.astype(np.float64) ** 2, axis=0))
    peak_values = np.max(np.abs(audio), axis=0)
    best = int(np.argmax(rms_values))

    print(f"\n{'ch':>2}  {'rms':>9}  {'peak':>9}  level")
    for number, (rms, peak) in enumerate(zip(rms_values, peak_values, strict=True)):
        print(f"{number:>2}  {rms:>9.6f}  {peak:>9.6f}  {_level(float(rms))}")

    print()
    if overflows:
        print(f"input overflows: {overflows}")
    print(f"strongest channel: {best}")
    if channel >= open_channels:
        print(
            f"note: --input-channel {channel} is outside the {open_channels} opened channel(s); "
            "increase --input-channels if needed."
        )
    if rms_values[best] < 0.002:
        print(
            "All channels look silent/quiet. Check that the microphone is on input 1 or 2, "
            "raise the interface gain, enable 48V for a condenser mic, and make sure the "
            "correct input is selected."
        )
        print(
            "If another input is active, use --input-channel N and, if the device needs a "
            "fixed channel count, --input-channels N."
        )
    else:
        name = sd.query_devices(index)["name"]
        print(
            f'Try: danglish talk --input-device "{name}" --input-channel {best} '
            f"{'--full-duplex ' if channel != best else ''}--stats"
        )
    return best


def check_output(
    sd: Any,
    device: int | str | None = None,
    *,
    channels: int | None = None,
    channel: int | None = None,
    sample_rate: int | None = None,
) -> bool:
    index = resolve_device(sd, device, "output")
    info = sd.query_devices(index, "output")
    rate = choose_rate(info, sample_rate)
    open_channels = choose_output_channels(info, channels, channel)
    print(f"Output: {describe_device(sd, index, 'output')}")
    print(f"Playing a 440 Hz test tone on {open_channels} channel(s) at {rate} Hz ...", flush=True)

    tone = np.sin(2 * np.pi * 440 * np.arange(int(rate * 0.45)) / rate).astype(np.float32)
    tone *= np.hanning(tone.size).astype(np.float32) * 0.05
    if channel is None:
        data = np.repeat(tone[:, None], open_channels, axis=1)
    else:
        selected = channel_index(channel, open_channels)
        data = np.zeros((tone.size, open_channels), dtype=np.float32)
        data[:, selected] = tone

    try:
        with sd.OutputStream(
            device=index,
            samplerate=rate,
            channels=open_channels,
            dtype="float32",
            blocksize=max(1, int(rate * 0.03)),
        ) as stream:
            stream.write(data)
    except Exception as exc:
        print(f"could not play on output device {index}: {type(exc).__name__}: {exc}")
        return False

    print(
        "Test tone complete. If you heard nothing, check the interface's monitor/volume "
        "knob and the selected output channel."
    )
    return True
