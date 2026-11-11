from __future__ import annotations

import pytest

from danglish.audio.device import (
    channel_index,
    choose_input_channels,
    choose_output_channels,
    choose_rate,
    describe_device,
    resolve_device,
)


class FakeSD:
    def __init__(self) -> None:
        self.hostapis = [
            {
                "name": "ALSA",
                "default_input_device": 0,
                "default_output_device": 0,
            },
            {
                "name": "PulseAudio",
                "default_input_device": 4,
                "default_output_device": 5,
            },
        ]
        self.devices = [
            {
                "index": 0,
                "name": "default",
                "hostapi": 0,
                "max_input_channels": 2,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            },
            {
                "index": 1,
                "name": "alsa_input.usb-Universal_Audio_Volt_4_123.analog-surround-40",
                "hostapi": 1,
                "max_input_channels": 4,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "index": 2,
                "name": "alsa_output.usb-Universal_Audio_Volt_4_123.analog-surround-40",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 4,
                "default_samplerate": 48000.0,
            },
            {
                "index": 3,
                "name": "alsa_output.usb-Universal_Audio_Volt_4_123.analog-surround-40.monitor",
                "hostapi": 1,
                "max_input_channels": 4,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "index": 4,
                "name": "Default Source",
                "hostapi": 1,
                "max_input_channels": 32,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "index": 5,
                "name": "Default Sink",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 32,
                "default_samplerate": 48000.0,
            },
            {
                "index": 6,
                "name": "alsa_input.pci.Mic1",
                "hostapi": 1,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
        ]

    def query_hostapis(self, index: int | None = None):
        if index is None:
            return self.hostapis
        return self.hostapis[index]

    def query_devices(self, device: int | None = None, kind: str | None = None):
        if device is None and kind is None:
            return self.devices
        if device is None:
            default = 4 if kind == "input" else 5
            return self.devices[default]
        if isinstance(device, int):
            return self.devices[device]
        raise AssertionError(f"unexpected device query: {device!r}")


def test_prefers_pulse_default_devices() -> None:
    sd = FakeSD()
    assert resolve_device(sd, None, "input") == 4
    assert resolve_device(sd, None, "output") == 5


def test_explicit_index_and_numeric_string() -> None:
    sd = FakeSD()
    assert resolve_device(sd, 1, "input") == 1
    assert resolve_device(sd, "1", "input") == 1
    assert resolve_device(sd, "2", "output") == 2


def test_resolves_interface_by_stable_name() -> None:
    sd = FakeSD()
    assert resolve_device(sd, "Volt 4", "input") == 1
    assert resolve_device(sd, "Volt 4", "output") == 2
    assert resolve_device(sd, "volt_4", "input") == 1


def test_resolves_default_by_name() -> None:
    sd = FakeSD()
    assert resolve_device(sd, "Default Source", "input") == 4
    assert resolve_device(sd, "Default Sink", "output") == 5
    assert resolve_device(sd, "default", "input") == 4


def test_unknown_or_ambiguous_name_is_rejected() -> None:
    sd = FakeSD()
    with pytest.raises(ValueError, match="no input device matching"):
        resolve_device(sd, "Scarlett", "input")

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_device(sd, "alsa", "input")


def test_describe_device_includes_name_rate_and_hostapi() -> None:
    text = describe_device(FakeSD(), 1, "input")
    assert "Volt_4" in text
    assert "4 ch" in text
    assert "48000 Hz" in text
    assert "PulseAudio" in text


def test_channel_choices_for_multichannel_interface() -> None:
    info = {"max_input_channels": 4, "max_output_channels": 4, "default_samplerate": 48000}
    assert choose_input_channels(info, None, 0) == 2
    assert choose_input_channels(info, None, 2) == 3
    assert choose_input_channels(info, 4, 0) == 4
    assert choose_output_channels(info, None, None) == 2
    assert choose_output_channels(info, None, 2) == 3
    assert choose_output_channels(info, 4, None) == 4
    assert choose_output_channels({"max_output_channels": 1}, None, None) == 1
    assert choose_rate(info, None) == 48000
    assert choose_rate(info, 16000) == 16000


def test_channel_index_clamps() -> None:
    assert channel_index(None, 2) == 0
    assert channel_index(1, 2) == 1
    assert channel_index(5, 2) == 1
    assert channel_index(0, 0) == 0
