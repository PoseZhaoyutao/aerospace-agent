"""Versioned WebSocket gateway for browser run requests."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from .manager import AgentRuntimeManager
from .protocol import RunStartRequest, RunTerminalEvent


async def websocket_endpoint(websocket: WebSocket, manager: AgentRuntimeManager) -> None:
    await websocket.accept()
    await websocket.send_json({"schema_version": "1.0.0", "type": "connection.ready"})
    try:
        while True:
            payload: dict[str, Any] = await websocket.receive_json()
            try:
                request = RunStartRequest.model_validate(payload)
            except ValidationError as exc:
                await websocket.send_json(
                    RunTerminalEvent(
                        type="run.failed",
                        request_id=str(payload.get("request_id") or "invalid-request"),
                        thread_id=str(payload.get("thread_id") or "invalid-thread"),
                        status="error",
                        answer=str(exc)[:1024],
                    ).model_dump(mode="json")
                )
                continue

            await websocket.send_json(
                {
                    "schema_version": "1.0.0",
                    "type": "run.accepted",
                    "request_id": request.request_id,
                    "thread_id": request.thread_id,
                }
            )
            await websocket.send_json(
                {
                    "schema_version": "1.0.0",
                    "type": "run.started",
                    "request_id": request.request_id,
                    "thread_id": request.thread_id,
                }
            )
            event = await asyncio.to_thread(
                manager.run,
                request.thread_id,
                request_id=request.request_id,
                message=request.message,
            )
            await websocket.send_json(event.model_dump(mode="json"))
    except WebSocketDisconnect:
        return


__all__ = ["websocket_endpoint"]
