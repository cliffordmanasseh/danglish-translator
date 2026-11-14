"""Configuration loading and defaults.

Configuration is intentionally small and dependency-free. A TOML file can set the
model, streaming, and server options; command-line flags override it.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib


DEFAULT_MODEL_DIR = Path("models/opus-mt-da-en-int8")
ENV_CONFIG = "DANGLISH_CONFIG"
ENV_MODEL = "DANGLISH_MODEL"
ENV_BACKEND = "DANGLISH_BACKEND"
ENV_SERVER_URL = "DANGLISH_SERVER_URL"
ENV_BROADCAST = "DANGLISH_BROADCAST"
ENV_BROADCAST_LOCAL = "DANGLISH_BROADCAST_LOCAL"
ENV_BROADCAST_SSID = "DANGLISH_BROADCAST_SSID"

COMPUTE_TYPES = ("int8", "int8_float32", "float32", "bfloat16", "int16", "float16")


@dataclass(slots=True)
class ModelConfig:
    path: Path = field(default_factory=lambda: DEFAULT_MODEL_DIR)
    device: str = "cpu"
    compute_type: str = "int8"
    intra_threads: int = 0  # 0 = auto (capped at 4)
    inter_threads: int = 1
    beam_size: int = 2
    max_decoding_length: int = 256
    backend: str = "ct2"
    server_url: str = "http://127.0.0.1:18080"

    def validate(self) -> None:
        if self.backend not in ("ct2", "llama"):
            raise ConfigError("model.backend must be 'ct2' or 'llama'")
        if not self.server_url.startswith(("http://", "https://")):
            raise ConfigError("model.server_url must start with http:// or https://")
        if self.compute_type not in COMPUTE_TYPES:
            raise ConfigError(
                f"unknown compute_type {self.compute_type!r}; choose one of {', '.join(COMPUTE_TYPES)}"
            )
        if self.device not in ("cpu", "auto"):
            raise ConfigError("this build targets CPU; set model.device = 'cpu'")
        if self.beam_size < 1:
            raise ConfigError("model.beam_size must be at least 1")
        if self.max_decoding_length < 1:
            raise ConfigError("model.max_decoding_length must be positive")
        if self.intra_threads < 0 or self.inter_threads < 1:
            raise ConfigError("thread counts must be non-negative, with at least one inter thread")


@dataclass(slots=True)
class StreamConfig:
    min_partial_chars: int = 12
    partial_debounce_ms: int = 200
    max_pending_chars: int = 4000

    def validate(self) -> None:
        if self.min_partial_chars < 1:
            raise ConfigError("stream.min_partial_chars must be positive")
        if not 0 <= self.partial_debounce_ms <= 5000:
            raise ConfigError("stream.partial_debounce_ms must be between 0 and 5000")
        if self.max_pending_chars < self.min_partial_chars:
            raise ConfigError("stream.max_pending_chars must be >= stream.min_partial_chars")


@dataclass(slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8765

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ConfigError("server.port must be between 1 and 65535")


@dataclass(slots=True)
class TTSConfig:
    voice: str = "en_GB-alba-medium"

    def validate(self) -> None:
        if not self.voice.strip():
            raise ConfigError("tts.voice must not be empty")


@dataclass(slots=True)
class BroadcastConfig:
    """Local Wi-Fi audio broadcast settings.

    ``local=True`` starts the server directly on the LAN without changing the
    Wi-Fi configuration. ``local=False`` uses the production launcher, which
    creates the open hotspot and requires sudo.
    """

    enabled: bool = False
    local: bool = True
    server_ip: str = "auto"
    ssid: str = "Church_Live_Audio"
    http_port: int = 8080
    https_port: int = 8443
    webrtc_port: int = 8555
    rtp_port: int = 5004
    audio_input: str = "danglish_broadcast.monitor"
    sink_name: str = "danglish_broadcast"

    def validate(self) -> None:
        if not self.server_ip.strip():
            raise ConfigError("broadcast.server_ip must not be empty")
        if not self.ssid.strip():
            raise ConfigError("broadcast.ssid must not be empty")
        if not self.audio_input.strip():
            raise ConfigError("broadcast.audio_input must not be empty")
        if not self.sink_name.strip():
            raise ConfigError("broadcast.sink_name must not be empty")
        for name in ("http_port", "https_port", "webrtc_port", "rtp_port"):
            port = getattr(self, name)
            if not 1 <= port <= 65535:
                raise ConfigError(f"broadcast.{name} must be between 1 and 65535")


@dataclass(slots=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    broadcast: BroadcastConfig = field(default_factory=BroadcastConfig)

    def validate(self) -> None:
        self.model.validate()
        self.stream.validate()
        self.server.validate()
        self.tts.validate()
        self.broadcast.validate()


def _apply_table(config: AppConfig, table: str, values: object) -> None:
    target = getattr(config, table, None)
    if target is None:
        raise ConfigError(f"unknown configuration section [{table}]")
    if not isinstance(values, dict):
        raise ConfigError(f"[{table}] must be a table")

    for key, value in values.items():
        if not hasattr(target, key):
            raise ConfigError(f"unknown configuration key {table}.{key}")
        current = getattr(target, key)
        if isinstance(current, Path):
            value = Path(str(value))
        elif isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, int):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ConfigError(f"{table}.{key} must be an integer")
        elif isinstance(current, str) and not isinstance(value, str):
            raise ConfigError(f"{table}.{key} must be a string")
        setattr(target, key, value)


def _env_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load configuration from TOML, then apply environment overrides.

    If *path* is omitted, ``$DANGLISH_CONFIG`` is consulted, then ``./danglish.toml``.
    Missing default files are fine; defaults are used instead.
    """

    config = AppConfig()
    config_path: Path | None

    if path is None:
        env_path = os.environ.get(ENV_CONFIG)
        config_path = Path(env_path) if env_path else None
        if config_path is None:
            candidate = Path("danglish.toml")
            if candidate.is_file():
                config_path = candidate
    else:
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigError(f"configuration file not found: {config_path}")

    if config_path is not None:
        try:
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"could not read {config_path}: {exc}") from exc

        for table, values in data.items():
            _apply_table(config, table, values)

    env_model = os.environ.get(ENV_MODEL)
    if env_model:
        config.model.path = Path(env_model)
    env_backend = os.environ.get(ENV_BACKEND)
    if env_backend:
        config.model.backend = env_backend
    env_server = os.environ.get(ENV_SERVER_URL)
    if env_server:
        config.model.server_url = env_server
    if _env_bool(os.environ.get(ENV_BROADCAST)):
        config.broadcast.enabled = True
    if ENV_BROADCAST_LOCAL in os.environ:
        config.broadcast.local = _env_bool(os.environ.get(ENV_BROADCAST_LOCAL))
    env_ssid = os.environ.get(ENV_BROADCAST_SSID)
    if env_ssid:
        config.broadcast.ssid = env_ssid

    config.validate()
    return config
