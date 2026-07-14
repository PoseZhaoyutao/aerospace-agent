"""Synchronous gateways for the local aerospace MCP server.

The agent is deliberately synchronous at its public boundary.  The stdio
implementation owns a private event loop and uses the official MCP SDK client
on that loop, so callers may use it from either synchronous code or an active
asyncio event loop.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..config import MCPSettings
from ..schema import ToolCallRequest, ToolCallResponse


class MCPUnavailableError(RuntimeError):
    """Structured error raised when the configured MCP transport is unavailable."""

    code = "MCP_UNAVAILABLE"
    status = "tool_unavailable"

    def __init__(self, message: str, *, reason: str | None = None):
        self.reason = reason or message
        self.error = self.reason
        self.details = {"code": self.code, "status": self.status, "reason": self.reason}
        super().__init__(self.reason)


class MCPGateway(Protocol):
    @property
    def closed(self) -> bool: ...

    def list_tools(self) -> list[Any]: ...

    def call_tool(self, request: ToolCallRequest) -> ToolCallResponse: ...

    def close(self) -> None: ...


def _mcp_tool_type():
    """Import MCP types lazily so importing the agent remains lightweight."""

    from mcp.types import Tool  # type: ignore

    return Tool


def _definitions_by_name() -> dict[str, dict[str, Any]]:
    from aerospace_agent.mcp.tools import get_tool_definitions

    return {item["name"]: item for item in get_tool_definitions() if item.get("name")}


def _missing_required(definition: Mapping[str, Any] | None, arguments: Mapping[str, Any]) -> list[str]:
    if definition is None:
        return []
    schema = definition.get("inputSchema") or {}
    required = schema.get("required") or []
    return [str(field) for field in required if field not in arguments]


def _response_error(tool_name: str, status: str, error: str, started: float) -> ToolCallResponse:
    return ToolCallResponse(
        tool_name=tool_name,
        status=status,
        error=error,
        duration_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
    )


class InProcessMCPGateway:
    """Explicit, synchronous fallback that invokes the local registry directly."""

    def __init__(
        self,
        tools: Mapping[str, Callable[..., Any]],
        definitions: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self._tools = dict(tools)
        source = definitions if definitions is not None else _definitions_by_name().values()
        self._definitions = {str(item["name"]): dict(item) for item in source if item.get("name")}
        # Test doubles and explicitly injected local tools may intentionally
        # have no published schema.  Give those handlers a permissive object
        # schema while retaining strict schemas for all official tools.
        for name in self._tools:
            self._definitions.setdefault(
                name,
                {
                    "name": name,
                    "description": "Injected local MCP tool",
                    "inputSchema": {"type": "object"},
                },
            )
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def list_tools(self) -> list[Any]:
        Tool = _mcp_tool_type()
        return [
            Tool(
                name=definition["name"],
                description=definition.get("description", ""),
                inputSchema=definition.get("inputSchema", {"type": "object"}),
            )
            for definition in self._definitions.values()
        ]

    def call_tool(self, request: ToolCallRequest) -> ToolCallResponse:
        started = time.perf_counter()
        if self._closed:
            return _response_error(request.tool_name, "tool_unavailable", "MCP gateway is closed", started)
        arguments = request.arguments or {}
        definition = self._definitions.get(request.tool_name)
        handler = self._tools.get(request.tool_name)
        if handler is None or definition is None:
            return _response_error(
                request.tool_name,
                "tool_unavailable",
                f"unknown MCP tool: {request.tool_name}",
                started,
            )
        missing = _missing_required(definition, arguments)
        if missing:
            return _response_error(
                request.tool_name,
                "invalid_arguments",
                "missing required arguments: " + ", ".join(missing),
                started,
            )
        try:
            result = handler(**arguments)
            if isinstance(result, Mapping) and result.get("status") not in (None, "success"):
                status = str(result.get("status"))
                error = result.get("error") or result.get("reason") or status
                return ToolCallResponse(
                    tool_name=request.tool_name,
                    status=status,
                    result=result,
                    error=str(error),
                    duration_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
                )
            return ToolCallResponse(
                tool_name=request.tool_name,
                status="success",
                result=result,
                duration_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
            )
        except Exception as exc:
            return _response_error(request.tool_name, "error", str(exc), started)

    def close(self) -> None:
        self._closed = True


class StdioMCPGateway:
    """Synchronous facade over the official MCP stdio client."""

    def __init__(
        self,
        command: str,
        args: Sequence[str] | None = None,
        *,
        timeout: float = 30.0,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.command = command
        self.args = list(args or [])
        self.timeout = float(timeout)
        self.env = dict(env) if env is not None else None
        self.cwd = cwd
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session: Any = None
        self._stdio_context: Any = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(target=self._thread_main, name="mcp-stdio-gateway", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=self.timeout):
            self._closed = True
            raise MCPUnavailableError("MCP stdio startup timed out")
        if self._startup_error is not None:
            self._closed = True
            error = self._startup_error
            if isinstance(error, MCPUnavailableError):
                raise error
            raise MCPUnavailableError(f"MCP stdio unavailable: {error}", reason=str(error)) from error

    @property
    def closed(self) -> bool:
        return self._closed

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._lifecycle())
        except BaseException as exc:  # propagate startup failures to constructor
            self._startup_error = exc
            self._ready.set()
        finally:
            self._loop.close()

    async def _lifecycle(self) -> None:
        from mcp import ClientSession  # type: ignore
        from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore

        parameters = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
            cwd=self.cwd,
        )
        self._stdio_context = stdio_client(parameters)
        streams = await self._stdio_context.__aenter__()
        try:
            self._session = ClientSession(*streams)
            await self._session.__aenter__()
            await self._session.initialize()
            self._stop_event = asyncio.Event()
            self._ready.set()
            await self._stop_event.wait()
        finally:
            if self._session is not None:
                try:
                    await self._session.__aexit__(None, None, None)
                except Exception:
                    pass
                self._session = None
            if self._stdio_context is not None:
                try:
                    await self._stdio_context.__aexit__(None, None, None)
                except Exception:
                    pass
                self._stdio_context = None

    def _run(self, coroutine: Any, *, timeout: float | None = None) -> Any:
        if self._closed or self._loop is None or not self._thread.is_alive():
            raise MCPUnavailableError("MCP gateway is closed")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=self.timeout if timeout is None else timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            self._abort_timed_out_request()
            raise TimeoutError("MCP stdio request timed out") from exc

    def _abort_timed_out_request(self) -> None:
        """Make a timed-out transport unusable and request lifecycle teardown.

        Cancellation of an MCP coroutine cannot prove that the child process
        stopped its underlying operation.  Closing the session boundary avoids
        reusing an indeterminate transport and lets the stdio context tear down
        its subprocess as soon as the event loop regains control.
        """
        self._closed = True
        loop = self._loop
        stop_event = self._stop_event
        if loop is not None and loop.is_running() and stop_event is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._set_stop(stop_event), loop)
            except Exception:
                pass

    def list_tools(self) -> list[Any]:
        if self._closed:
            return []
        try:
            result = self._run(self._session.list_tools())
            return list(result.tools)
        except Exception as exc:
            if isinstance(exc, MCPUnavailableError):
                raise
            raise MCPUnavailableError(f"MCP stdio list_tools failed: {exc}", reason=str(exc)) from exc

    def call_tool(self, request: ToolCallRequest) -> ToolCallResponse:
        started = time.perf_counter()
        if self._closed:
            return _response_error(request.tool_name, "tool_unavailable", "MCP gateway is closed", started)
        try:
            result = self._run(
                self._session.call_tool(request.tool_name, request.arguments),
                timeout=min(self.timeout, request.timeout_seconds),
            )
            payload: Any = None
            text_parts: list[str] = []
            for content in getattr(result, "content", []) or []:
                text = getattr(content, "text", None)
                if text is not None:
                    text_parts.append(text)
            if text_parts:
                import json

                raw = "\n".join(text_parts)
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = raw
            status = "error" if getattr(result, "isError", False) else "success"
            if isinstance(payload, dict) and payload.get("status") not in (None, "success"):
                status = str(payload["status"])
            if status != "success":
                return _response_error(
                    request.tool_name,
                    status,
                    str(payload.get("error") or payload.get("reason") or payload or "MCP tool failed")
                    if isinstance(payload, dict)
                    else str(payload),
                    started,
                )
            return ToolCallResponse(
                tool_name=request.tool_name,
                status="success",
                result=payload,
                duration_ms=max(0.0, (time.perf_counter() - started) * 1000.0),
            )
        except TimeoutError as exc:
            return _response_error(request.tool_name, "timeout", str(exc), started)
        except Exception as exc:
            return _response_error(request.tool_name, "tool_unavailable", str(exc), started)

    def close(self) -> None:
        self._closed = True
        loop = self._loop
        stop_event = self._stop_event
        if loop is not None and loop.is_running() and stop_event is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._set_stop(stop_event), loop).result(
                    timeout=self.timeout
                )
            except Exception:
                pass
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=self.timeout)

    @staticmethod
    async def _set_stop(stop_event: asyncio.Event) -> None:
        stop_event.set()


def create_mcp_gateway(
    mcp_settings: MCPSettings,
    *,
    allow_inprocess_fallback: bool = False,
    force_stdio_failure: bool = False,
) -> tuple[MCPGateway, list[str]]:
    """Construct the configured gateway, with an explicit opt-in fallback."""

    warnings: list[str] = []
    try:
        if mcp_settings.transport != "stdio":
            raise MCPUnavailableError(f"unsupported MCP transport: {mcp_settings.transport}")
        if force_stdio_failure:
            raise MCPUnavailableError("MCP stdio unavailable (forced failure)")
        gateway: MCPGateway = StdioMCPGateway(
            command=mcp_settings.command,
            args=mcp_settings.args,
        )
        return gateway, warnings
    except Exception as exc:
        if not allow_inprocess_fallback:
            if isinstance(exc, MCPUnavailableError):
                raise
            raise MCPUnavailableError(f"MCP stdio unavailable: {exc}", reason=str(exc)) from exc
        from aerospace_agent.mcp.server import _wrap_all_tools

        warnings.append("MCP stdio unavailable; using explicit in-process fallback")
        return InProcessMCPGateway(_wrap_all_tools()), warnings


__all__ = [
    "MCPGateway",
    "MCPUnavailableError",
    "InProcessMCPGateway",
    "StdioMCPGateway",
    "create_mcp_gateway",
]
