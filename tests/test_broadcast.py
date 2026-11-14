from __future__ import annotations

from pathlib import Path

from danglish.broadcast import BroadcastServer
from danglish.config import BroadcastConfig


def test_local_command_uses_the_translated_sink(tmp_path: Path) -> None:
    config = BroadcastConfig(
        enabled=True,
        local=True,
        server_ip="127.0.0.1",
        http_port=18080,
        audio_input="danglish_broadcast.monitor",
    )
    server = BroadcastServer(config, directory=tmp_path)
    command = server._command()

    assert command[0].endswith("church-audio-server")
    assert "--audio-input=danglish_broadcast.monitor" in command
    assert "--ip=127.0.0.1" in command
    assert server.listener_url == "http://127.0.0.1:18080/"
    assert server.qr_url == "http://127.0.0.1:18080/qr"


def test_hotspot_uses_the_production_launcher(tmp_path: Path) -> None:
    config = BroadcastConfig(
        enabled=True,
        local=False,
        server_ip="10.42.0.1",
        ssid="Church_Live_Audio",
    )
    server = BroadcastServer(config, directory=tmp_path)

    assert server._command() == ["bash", str(tmp_path / "start_church_broadcast.sh")]
    assert server.listener_url == "http://10.42.0.1:8080/"
