"""MCP Server 入口 — 统一航天动力学 MCP 服务器。

第一性原理（K2 服务边界）：
  1. 所有工具调用包裹在 error_handler 中——绝不静默失败
  2. 启动时打印引擎可用性，供运维和 LLM 决策
  3. mcp 包可用时启动标准 MCP Server；否则打印可用工具列表
  4. 每个工具的异常都转为结构化 {status:"error", reason:...}
"""
from __future__ import annotations

import functools
import json
import sys
import traceback
from typing import Any, Callable, Dict

from .adapters import get_all_adapters
from .tools import TOOL_REGISTRY, get_tool_definitions, CORE_TOOLS
from .tools.environment_tools import check_engine_availability


def error_handler(func: Callable[..., Dict]) -> Callable[..., Dict]:
    """工具调用统一错误处理装饰器。

    捕获所有异常，转为结构化 {status:"error", reason:..., traceback:...}。
    绝不静默失败。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Dict:
        try:
            result = func(*args, **kwargs)
            if result is None:
                return {
                    "status": "error",
                    "reason": "工具返回 None——内部逻辑异常",
                    "tool": func.__name__,
                }
            # 确保结果是可序列化的
            _ensure_serializable(result)
            return result
        except TypeError as exc:
            return {
                "status": "error",
                "reason": f"参数类型错误: {exc}",
                "tool": func.__name__,
            }
        except ValueError as exc:
            return {
                "status": "error",
                "reason": f"数值错误: {exc}",
                "tool": func.__name__,
            }
        except Exception as exc:
            return {
                "status": "error",
                "reason": f"未预期错误: {exc}",
                "tool": func.__name__,
                "traceback": traceback.format_exc().split("\n")[-5:],
            }
    return wrapper


def _ensure_serializable(obj: Any) -> None:
    """验证对象是否 JSON 可序列化（不可序列化时抛出 ValueError）。"""
    try:
        json.dumps(obj, default=str)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"工具返回值不可 JSON 序列化: {exc}")


def _wrap_all_tools() -> Dict[str, Callable[..., Dict]]:
    """用 error_handler 包装所有注册的工具。"""
    return {name: error_handler(fn) for name, fn in TOOL_REGISTRY.items()}


def print_startup_info() -> None:
    """启动时打印引擎可用性和工具列表。"""
    print("=" * 70)
    print("  astro_dynamics_mcp — 统一航天动力学 MCP Server")
    print("=" * 70)
    print()

    # 引擎可用性
    print("[引擎可用性检查]")
    availability = check_engine_availability()
    for engine, info in availability.items():
        status = "可用" if info.get("available") else "不可用"
        version = info.get("version", "N/A")
        caps = info.get("capabilities", [])
        caps_str = ", ".join(caps) if caps else "无"
        reason = info.get("reason", "")
        line = f"  {engine:12s} | {status:4s} | v{version:20s} | 能力: {caps_str}"
        if reason:
            line += f"\n               | 原因: {reason}"
        print(line)

    print()
    print(f"[已注册工具] 共 {len(CORE_TOOLS)} 个核心工具:")
    for i, tool_name in enumerate(CORE_TOOLS, 1):
        print(f"  {i:2d}. {tool_name}")

    print()
    available_count = sum(1 for v in availability.values()
                          if v.get("available"))
    print(f"  可用引擎: {available_count} / {len(availability)}")
    print("=" * 70)


def main() -> None:
    """MCP Server 主入口。

    优先使用 mcp 包启动标准 MCP 服务器；
    若 mcp 包不可用，则打印可用工具列表并进入命令行模式。
    """
    print_startup_info()

    wrapped_tools = _wrap_all_tools()

    # 尝试启动 MCP Server
    try:
        from mcp.server import Server  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
        _run_mcp_server(wrapped_tools)
        return
    except ImportError:
        print("\n[mcp 包未安装] 启动命令行模式。")
        print("安装 MCP 支持: pip install 'astro-dynamics-mcp[mcp]'")
        print()
        _run_cli_mode(wrapped_tools)
    except Exception as exc:
        print(f"\n[MCP Server 启动失败: {exc}]")
        print("回退到命令行模式。\n")
        _run_cli_mode(wrapped_tools)


def _run_mcp_server(tools: Dict[str, Callable]) -> None:
    """使用 mcp 包启动标准 MCP 服务器。"""
    from mcp.server import Server  # type: ignore
    from mcp.types import Tool  # type: ignore

    server = Server("astro-dynamics-mcp")
    definitions = get_tool_definitions()

    @server.list_tools()
    async def list_tools() -> list:
        return [
            Tool(
                name=d["name"],
                description=d["description"],
                inputSchema=d["inputSchema"],
            )
            for d in definitions
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
        if name not in tools:
            return [{"type": "text", "text": json.dumps({
                "status": "error",
                "reason": f"未知工具: {name}",
            })}]
        result = tools[name](**arguments)
        return [{"type": "text", "text": json.dumps(result, default=str)}]

    import asyncio

    async def run():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    print("\n[MCP Server 已启动] 等待工具调用...\n")
    asyncio.run(run())


def _run_cli_mode(tools: Dict[str, Callable]) -> None:
    """命令行交互模式（mcp 包不可用时的回退）。"""
    print("可用命令:")
    print("  <tool_name> <json_args>  — 调用工具")
    print("  list                      — 列出所有工具")
    print("  defs                      — 打印工具定义")
    print("  quit                      — 退出")
    print()

    while True:
        try:
            line = input("astro_dynamics> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
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
        args = {}
        if len(parts) > 1:
            try:
                args = json.loads(parts[1])
            except json.JSONDecodeError as exc:
                print(f"参数 JSON 解析失败: {exc}")
                continue

        if tool_name not in tools:
            print(f"未知工具: {tool_name}。输入 'list' 查看可用工具。")
            continue

        result = tools[tool_name](**args)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
