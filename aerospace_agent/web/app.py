"""FastAPI application factory for the local WebUI gateway."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .manager import AgentRuntimeManager, ThreadScopeError
from .protocol import (
    HealthResponse,
    RuntimeStatusResponse,
    ThreadCreateRequest,
    ThreadListResponse,
    ThreadSummary,
)
from .websocket import websocket_endpoint


def _not_found(exc: ThreadScopeError) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"code": "THREAD_NOT_FOUND", "message": str(exc)},
    )


def create_app(
    *,
    manager: AgentRuntimeManager,
    static_dir: str | Path | None = None,
    webui_enabled: bool = True,
    host: str = "127.0.0.1",
    allow_lan: bool = False,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager.start()
        app.state.webui_manager = manager
        try:
            yield
        finally:
            manager.shutdown()

    app = FastAPI(title="Aerospace Agent WebUI", version="1.0.0", lifespan=lifespan)
    static_root = Path(static_dir or Path(__file__).with_name("static"))
    loopback = host in {"127.0.0.1", "localhost", "::1"}

    @app.get("/")
    async def frontend_root():
        if not webui_enabled:
            raise HTTPException(
                status_code=503,
                detail={"code": "WEBUI_DISABLED", "message": "WebUI is disabled"},
            )
        if allow_lan or not loopback:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "WEBUI_LAN_NOT_ALLOWED",
                    "message": "LAN WebUI access is not authenticated in phase one",
                },
            )
        index = static_root / "index.html"
        if not index.is_file():
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "WEBUI_STATIC_NOT_BUILT",
                    "message": "WebUI static assets are not built",
                },
            )
        return FileResponse(index)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Any, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and "code" in detail:
            payload = {"schema_version": "1.0.0", "error": detail}
        else:
            payload = {
                "schema_version": "1.0.0",
                "error": {"code": "HTTP_ERROR", "message": str(detail)},
            }
        return JSONResponse(status_code=exc.status_code, content=payload)

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ready")

    @app.get("/api/v1/runtime", response_model=RuntimeStatusResponse)
    async def runtime() -> RuntimeStatusResponse:
        agent = manager.agent if manager._started else None
        return RuntimeStatusResponse(
            model=str(getattr(agent, "model_name", "unknown") or "unknown"),
            connection="connected" if agent is not None else "disconnected",
            active_runs=manager.active_runs,
        )

    @app.get("/api/v1/threads", response_model=ThreadListResponse)
    async def list_threads() -> ThreadListResponse:
        return ThreadListResponse(threads=manager.list_threads())

    @app.post("/api/v1/threads", response_model=ThreadSummary, status_code=201)
    async def create_thread(request: ThreadCreateRequest) -> ThreadSummary:
        return manager.create_thread(title=request.title)

    @app.get("/api/v1/threads/{thread_id}", response_model=ThreadSummary)
    async def get_thread(thread_id: str) -> ThreadSummary:
        try:
            return manager.get_thread(thread_id)
        except ThreadScopeError as exc:
            raise _not_found(exc) from exc

    @app.get("/api/v1/threads/{thread_id}/history")
    async def history(thread_id: str):
        try:
            return manager.history(thread_id)
        except ThreadScopeError as exc:
            raise _not_found(exc) from exc

    async def websocket_handler(websocket: WebSocket):
        await websocket_endpoint(websocket, manager)

    app.add_api_websocket_route("/api/v1/ws", websocket_handler)

    assets = static_root / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="webui-assets")

    return app


__all__ = ["create_app"]
