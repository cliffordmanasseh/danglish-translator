from __future__ import annotations

import json
import urllib.error

import pytest

from danglish.backends.llama import LlamaServerBackend
from danglish.errors import BackendError


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_translate_sends_training_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        return FakeResponse({"content": "God is love."})

    monkeypatch.setattr("danglish.backends.llama.urllib.request.urlopen", fake_urlopen)
    backend = LlamaServerBackend("http://127.0.0.1:18080")
    assert backend.translate(["Gud er kærlighed."]) == ["God is love."]
    assert "Gud er kærlighed." in seen["body"]["prompt"]
    assert "English:" in seen["body"]["prompt"]
    assert seen["body"]["stop"] == ["<|im_end|>"]


def test_empty_text_is_not_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("urlopen should not be called for empty input")

    monkeypatch.setattr("danglish.backends.llama.urllib.request.urlopen", fail)
    backend = LlamaServerBackend()
    assert backend.translate(["   "]) == [""]


def test_server_error_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("danglish.backends.llama.urllib.request.urlopen", fail)
    backend = LlamaServerBackend()
    with pytest.raises(BackendError, match="llama.cpp server"):
        backend.translate(["Gud er kærlighed."])
