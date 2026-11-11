from __future__ import annotations

from fastapi.testclient import TestClient

from danglish.backends.mock import MockBackend
from danglish.config import AppConfig
from danglish.server.app import create_app


def _client() -> TestClient:
    config = AppConfig()
    config.stream.partial_debounce_ms = 30
    app = create_app(config, backend=MockBackend())
    return TestClient(app)


def _receive_until(websocket, kind: str, attempts: int = 20) -> dict[str, object]:
    for _ in range(attempts):
        message = websocket.receive_json()
        if message.get("type") == kind:
            return message
    raise AssertionError(f"no {kind!r} message received")


def test_index_page() -> None:
    with _client() as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Danglish" in response.text
        assert "Danish" in response.text


def test_health_and_info() -> None:
    with _client() as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        info = client.get("/api/info").json()
        assert info["name"] == "danglish"
        assert info["mock"] is True
        assert info["rss_mb"] > 0


def test_websocket_completes_a_sentence() -> None:
    with _client() as client, client.websocket_connect("/ws/translate") as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert ready["mock"] is True

        websocket.send_json({"type": "delta", "text": "Hej med dig. "})
        event = websocket.receive_json()
        assert event["type"] == "final"
        assert event["id"] == 1
        assert event["source"] == "Hej med dig."
        assert event["target"].startswith("[mock]")


def test_websocket_flush_and_partial() -> None:
    with _client() as client, client.websocket_connect("/ws/translate") as websocket:
        websocket.receive_json()  # ready

        websocket.send_json({"type": "delta", "text": "Hvordan går det"})
        websocket.send_json({"type": "flush"})
        event = _receive_until(websocket, "final")
        assert event["source"] == "Hvordan går det"


def test_websocket_reset() -> None:
    with _client() as client, client.websocket_connect("/ws/translate") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "delta", "text": "Hej. "})
        _receive_until(websocket, "final")
        websocket.send_json({"type": "reset"})
        assert _receive_until(websocket, "reset")["type"] == "reset"


def test_websocket_rejects_bad_messages() -> None:
    with _client() as client, client.websocket_connect("/ws/translate") as websocket:
        websocket.receive_json()
        websocket.send_text("{not-json")
        error = websocket.receive_json()
        assert error["type"] == "error"

        websocket.send_json({"type": "nonsense"})
        error = websocket.receive_json()
        assert error["type"] == "error"

        websocket.send_json({"type": "delta", "text": 42})
        error = websocket.receive_json()
        assert error["type"] == "error"


def test_websocket_ping() -> None:
    with _client() as client, client.websocket_connect("/ws/translate") as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json()["type"] == "pong"


def test_client_session_info_has_version() -> None:
    with _client() as client:
        payload = client.get("/api/info").json()
        assert "version" in payload
