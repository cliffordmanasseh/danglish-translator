"""FastAPI WebSocket server and local web demo."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .. import __version__
from ..backends import create_backend
from ..backends.base import TranslatorBackend
from ..config import AppConfig, load_config
from ..context import ContextProfile, get_profile
from ..memory import current_rss_mb
from ..pipeline import StreamingPipeline

logger = logging.getLogger("danglish.server")
_STATIC_DIR = Path(__file__).with_name("static")


class ClientSession:
    """One streaming translation session over a WebSocket."""

    def __init__(
        self,
        websocket: WebSocket,
        pipeline: StreamingPipeline,
        *,
        debounce_ms: int,
        info: dict[str, Any],
    ) -> None:
        self._ws = websocket
        self._pipeline = pipeline
        self._debounce = max(0, debounce_ms) / 1000.0
        self._info = info
        self._version = 0
        self._closed = False
        self._translate_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        await self.send({"type": "ready", **self._info})

    async def send(self, payload: dict[str, Any]) -> bool:
        async with self._send_lock:
            if self._closed:
                return False
            try:
                await self._ws.send_json(payload)
            except (WebSocketDisconnect, RuntimeError):
                self._closed = True
                return False
        return True

    async def close(self) -> None:
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def handle_raw(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            await self.send({"type": "error", "message": "invalid JSON"})
            return

        if not isinstance(message, dict):
            await self.send({"type": "error", "message": "expected a JSON object"})
            return

        message_type = message.get("type")
        if message_type == "delta":
            await self._handle_delta(message.get("text"))
        elif message_type == "flush":
            await self._handle_flush()
        elif message_type == "reset":
            await self._handle_reset()
        elif message_type == "ping":
            await self.send({"type": "pong"})
        else:
            await self.send({"type": "error", "message": f"unknown message type: {message_type}"})

    async def _handle_delta(self, text: object) -> None:
        if not isinstance(text, str):
            await self.send({"type": "error", "message": "delta.text must be a string"})
            return
        if not text:
            return

        self._version += 1
        version = self._version
        async with self._translate_lock:
            events = await asyncio.to_thread(self._pipeline.push, text)
        await self._emit(events)

        if self._pipeline.pending:
            self._schedule_preview(version)

    async def _handle_flush(self) -> None:
        self._version += 1
        async with self._translate_lock:
            events = await asyncio.to_thread(self._pipeline.flush)
        await self._emit(events)

    async def _handle_reset(self) -> None:
        self._version += 1
        async with self._translate_lock:
            self._pipeline.reset()
        await self.send({"type": "reset"})

    async def _emit(self, events: list[Any]) -> None:
        for event in events:
            await self.send(event.to_dict())

    def _schedule_preview(self, version: int) -> None:
        task = asyncio.create_task(self._preview_after(version))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _preview_after(self, version: int) -> None:
        try:
            await asyncio.sleep(self._debounce)
            if self._closed or version != self._version:
                return
            async with self._translate_lock:
                if self._closed or version != self._version:
                    return
                event = await asyncio.to_thread(self._pipeline.preview)
            if event is not None and not self._closed and version == self._version:
                await self.send(event.to_dict())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("partial translation failed")


def create_app(
    config: AppConfig | None = None,
    *,
    backend: TranslatorBackend | None = None,
    mock: bool = False,
    context: ContextProfile | str = "general",
) -> FastAPI:
    cfg = config or load_config()
    cfg.validate()
    context_profile = get_profile(context) if isinstance(context, str) else context

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if backend is not None:
            app.state.backend = backend
        else:
            app.state.backend = await asyncio.to_thread(create_backend, cfg, mock=mock)
        app.state.backend_name = getattr(app.state.backend, "name", "unknown")
        try:
            yield
        finally:
            close = getattr(app.state.backend, "close", None)
            if close is not None:
                await asyncio.to_thread(close)

    app = FastAPI(
        title="Danglish",
        version=__version__,
        description="Local, streaming Danish-to-English translation.",
        lifespan=lifespan,
    )
    app.state.config = cfg
    app.state.context = context_profile
    app.state.mock = mock or (backend is not None and getattr(backend, "name", "") == "mock")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/info")
    async def api_info() -> dict[str, Any]:
        active_backend = getattr(app.state, "backend", None)
        return {
            "name": "danglish",
            "version": __version__,
            "backend": getattr(active_backend, "name", None),
            "mock": bool(app.state.mock),
            "model": "mock" if app.state.mock else str(cfg.model.path),
            "compute_type": cfg.model.compute_type,
            "context": app.state.context.name,
            "rss_mb": round(current_rss_mb(), 1),
        }

    @app.websocket("/ws/translate")
    async def translate_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        pipeline = StreamingPipeline(
            app.state.backend,
            min_partial_chars=cfg.stream.min_partial_chars,
            max_pending_chars=cfg.stream.max_pending_chars,
            context=app.state.context,
        )
        info = {
            "model": "mock" if app.state.mock else Path(cfg.model.path).name,
            "mock": bool(app.state.mock),
            "compute_type": cfg.model.compute_type,
            "context": app.state.context.name,
            "rss_mb": round(current_rss_mb(), 1),
        }
        session = ClientSession(
            websocket,
            pipeline,
            debounce_ms=cfg.stream.partial_debounce_ms,
            info=info,
        )
        await session.start()
        try:
            while True:
                raw = await websocket.receive_text()
                await session.handle_raw(raw)
        except WebSocketDisconnect:
            pass
        except RuntimeError:
            pass
        finally:
            await session.close()

    return app


def run_server(
    config: AppConfig,
    *,
    mock: bool = False,
    context: ContextProfile | str = "general",
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - guarded by the CLI as well
        raise RuntimeError('the server extra is missing; install "danglish[server]"') from exc

    if config.server.host not in {"127.0.0.1", "localhost", "::1"}:
        logger.warning(
            "binding to %s exposes the demo to your network; there is no authentication",
            config.server.host,
        )

    app = create_app(config, mock=mock, context=context)
    print(
        f"Danglish {__version__} — http://{config.server.host}:{config.server.port}",
        flush=True,
    )
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info",
    )
