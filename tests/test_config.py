from __future__ import annotations

from pathlib import Path

import pytest

from danglish.config import AppConfig, load_config
from danglish.errors import ConfigError


def test_defaults_are_valid() -> None:
    config = AppConfig()
    config.validate()
    assert config.model.compute_type == "int8"
    assert config.model.beam_size == 2
    assert config.server.port == 8765
    assert config.tts.voice == "en_GB-alba-medium"


def test_load_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[model]
path = "custom/model"
compute_type = "float32"
intra_threads = 2
beam_size = 1

[stream]
min_partial_chars = 20
partial_debounce_ms = 50

[server]
host = "0.0.0.0"
port = 9000

[tts]
voice = "en_GB-cori-high"
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.path == Path("custom/model")
    assert config.model.compute_type == "float32"
    assert config.model.intra_threads == 2
    assert config.model.beam_size == 1
    assert config.stream.min_partial_chars == 20
    assert config.stream.partial_debounce_ms == 50
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 9000
    assert config.tts.voice == "en_GB-cori-high"


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[nonsense]\nvalue = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown configuration section"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[model]\nbeam = 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config(path)


def test_invalid_compute_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[model]\ncompute_type = "quantum"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown compute_type"):
        load_config(path)


def test_wrong_type_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[model]\nbeam_size = "four"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="must be an integer"):
        load_config(path)


def test_missing_explicit_config_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.toml")


def test_environment_model_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[model]\npath = "from-file"\n', encoding="utf-8")
    monkeypatch.setenv("DANGLISH_MODEL", "/tmp/from-env")

    config = load_config(path)

    assert config.model.path == Path("/tmp/from-env")


def test_no_config_file_uses_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DANGLISH_CONFIG", raising=False)
    monkeypatch.delenv("DANGLISH_MODEL", raising=False)

    config = load_config()

    assert config.model.path == Path("models/opus-mt-da-en-int8")


def test_invalid_port() -> None:
    config = AppConfig()
    config.server.port = 0
    with pytest.raises(ConfigError, match="server.port"):
        config.validate()


def test_max_pending_must_exceed_partial_minimum() -> None:
    config = AppConfig()
    config.stream.min_partial_chars = 50
    config.stream.max_pending_chars = 10
    with pytest.raises(ConfigError, match="max_pending_chars"):
        config.validate()


def test_llama_backend_defaults_and_validation() -> None:
    config = AppConfig()
    assert config.model.backend == "ct2"
    assert config.model.server_url == "http://127.0.0.1:18080"

    config.model.backend = "llama"
    config.validate()

    config.model.backend = "invalid"
    with pytest.raises(ConfigError, match="model.backend"):
        config.validate()


def test_load_toml_llama_backend(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[model]
backend = "llama"
server_url = "http://127.0.0.1:9999"
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.backend == "llama"
    assert config.model.server_url == "http://127.0.0.1:9999"


def test_broadcast_defaults() -> None:
    config = AppConfig()
    assert config.broadcast.enabled is False
    assert config.broadcast.local is True
    assert config.broadcast.server_ip == "auto"
    assert config.broadcast.audio_input == "danglish_broadcast.monitor"
    assert config.broadcast.http_port == 8080


def test_broadcast_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[broadcast]
enabled = true
local = false
ssid = "Church_Test"
http_port = 9090
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.broadcast.enabled is True
    assert config.broadcast.local is False
    assert config.broadcast.ssid == "Church_Test"
    assert config.broadcast.http_port == 9090


def test_broadcast_invalid_port() -> None:
    config = AppConfig()
    config.broadcast.http_port = 0
    with pytest.raises(ConfigError, match="broadcast.http_port"):
        config.validate()


def test_broadcast_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DANGLISH_CONFIG", raising=False)
    monkeypatch.setenv("DANGLISH_BROADCAST", "1")
    monkeypatch.setenv("DANGLISH_BROADCAST_LOCAL", "0")
    monkeypatch.setenv("DANGLISH_BROADCAST_SSID", "Church_Env")

    config = load_config()

    assert config.broadcast.enabled is True
    assert config.broadcast.local is False
    assert config.broadcast.ssid == "Church_Env"
