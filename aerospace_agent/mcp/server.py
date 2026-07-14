"""Protocol-clean entry point for the aerospace MCP server."""
from __future__ import annotations

import functools
import json
import sys
import traceback
from typing import Any, Callable, Dict

from .tools import CORE_TOOLS, TOOL_REGISTRY, get_tool_definitions
from .tools.environment_tools import check_engine_availability


def error_handler(func: Callable[..., Dict]) -> Callable[..., Dict]:
    """Convert every tool failure into a JSON-serializable result."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Dict:
        try:
            result = func(*args, **kwargs)
            if result is None:
                return {"status": "error", "reason": "tool returned None", "tool": func.__name__}
            _ensure_serializable(result)
            return result
        except TypeError as exc:
            return {"status": "error", "reason": f"argument type error: {exc}", "tool": func.__name__}
        except ValueError as exc:
            return {"status": "error", "reason": f"value error: {exc}", "tool": func.__name__}
        except Exception as exc:
            return {
                "status": "error",
                "reason": f"unexpected error: {exc}",
                "tool": func.__name__,
                "traceback": traceback.format_exc().split("\n")[-5:],
            }

    return wrapper


def _ensure_serializable(obj: Any) -> None:
    try:
        json.dumps(obj, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tool result is not JSON serializable: {exc}") from exc


def _wrap_all_tools() -> Dict[str, Callable[..., Dict]]:
    # ``get_tool_definitions`` lazily registers research tools.  Populate the
    # registry before taking its snapshot so every advertised tool has a
    # callable handler.
    get_tool_definitions()
    return {name: error_handler(fn) for name, fn in TOOL_REGISTRY.items()}


def print_startup_info(*, file=None, include_availability: bool = True) -> None:
    """Print human diagnostics, never protocol frames.

    ``include_availability=False`` avoids probing optional/commercial engines
    during stdio startup.  In stdio mode all output is directed to stderr.
    """

    if file is None:
        file = sys.stdout
    print("=" * 70, file=file)
    print("  astro_dynamics_mcp MCP Server", file=file)
    print("=" * 70, file=file)
    if include_availability:
        print("[engine availability]", file=file)
        availability = check_engine_availability()
        for engine, info in availability.items():
            status = "available" if info.get("available") else "unavailable"
            version = info.get("version", "N/A")
            caps = ", ".join(info.get("capabilities", [])) or "none"
            reason = info.get("reason", "")
            line = f"  {engine:12s} | {status:11s} | v{version:20s} | capabilities: {caps}"
            if reason:
                line += f"\n               | reason: {reason}"
            print(line, file=file)
    print(file=file)
    print(f"[registered tools] {len(CORE_TOOLS)} core tools", file=file)
    for i, tool_name in enumerate(CORE_TOOLS, 1):
        print(f"  {i:2d}. {tool_name}", file=file)
    if include_availability:
        count = sum(1 for info in availability.values() if info.get("available"))
        print(f"  available engines: {count} / {len(availability)}", file=file)
    print("=" * 70, file=file)


def main() -> None:
    wrapped_tools = _wrap_all_tools()
    try:
        from mcp.server import Server  # type: ignore  # noqa: F401
    except ImportError:
        print_startup_info()
        print("\n[mcp package unavailable] entering CLI mode")
        print("Install MCP support with: pip install 'astro-dynamics-mcp[mcp]'\n")
        _run_cli_mode(wrapped_tools)
        return

    # stdout is reserved for JSON-RPC frames in stdio mode.
    print_startup_info(file=sys.stderr, include_availability=False)
    _run_mcp_server(wrapped_tools)


def _run_mcp_server(tools: Dict[str, Callable]) -> None:
    """Run the official MCP stdio server with protocol-safe content types."""

    from mcp.server import Server  # type: ignore
    from mcp.server.stdio import stdio_server  # type: ignore
    from mcp.types import TextContent, Tool  # type: ignore

    server = Server("astro-dynamics-mcp")
    definitions = get_tool_definitions()

    @server.list_tools()
    async def list_tools() -> list:
        return [
            Tool(
                name=definition["name"],
                description=definition["description"],
                inputSchema=definition["inputSchema"],
            )
            for definition in definitions
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list:
        arguments = arguments or {}
        if name not in tools:
            result = {"status": "tool_unavailable", "error": f"unknown MCP tool: {name}"}
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        definition = next((item for item in definitions if item.get("name") == name), None)
        required = ((definition or {}).get("inputSchema") or {}).get("required", [])
        missing = [field for field in required if field not in arguments]
        if missing:
            result = {
                "status": "invalid_arguments",
                "error": "missing required arguments: " + ", ".join(missing),
                "tool": name,
            }
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        result = tools[name](**arguments)
        return [TextContent(type="text", text=json.dumps(result, default=str, ensure_ascii=False))]

    import asyncio

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    print("[MCP Server started; waiting for tool calls]", file=sys.stderr)
    asyncio.run(run())


def _run_cli_mode(tools: Dict[str, Callable]) -> None:
    print("Available commands:")
    print("  <tool_name> <json_args>  call a tool")
    print("  list                     list tools")
    print("  defs                     print tool definitions")
    print("  quit                     exit")
    print()
    while True:
        try:
            line = input("astro_dynamics> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        if line == "list":
            for name in CORE_TOOLS:
                print(f"  {name}")
            continue
        if line == "defs":
            print(json.dumps(get_tool_definitions(), indent=2, ensure_ascii=False))
            continue
        parts = line.split(None, 1)
        tool_name = parts[0]
        args: dict[str, Any] = {}
        if len(parts) > 1:
            try:
                args = json.loads(parts[1])
            except json.JSONDecodeError as exc:
                print(f"JSON argument parse failed: {exc}")
                continue
        if tool_name not in tools:
            print(f"Unknown tool: {tool_name}")
            continue
        print(json.dumps(tools[tool_name](**args), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
