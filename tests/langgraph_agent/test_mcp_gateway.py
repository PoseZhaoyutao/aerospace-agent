from __future__ import annotations

import asyncio
import sys

import pytest

from aerospace_agent.langgraph_agent.schema import ToolCallRequest
from aerospace_agent.langgraph_agent.services.mcp_gateway import (
    InProcessMCPGateway,
    MCPUnavailableError,
    StdioMCPGateway,
    create_mcp_gateway,
)
from aerospace_agent.mcp.server import _wrap_all_tools
from aerospace_agent.mcp.tool_schema import get_tool_definition_report
from aerospace_agent.mcp.tools import get_tool_definitions


def test_inprocess_gateway_lists_and_calls_schema_valid_tools():
    gateway = InProcessMCPGateway(_wrap_all_tools())
    names = {tool.name for tool in gateway.list_tools()}
    assert "check_engine_availability" in names
    response = gateway.call_tool(ToolCallRequest(tool_name="check_engine_availability"))
    assert response.status == "success"


def test_missing_required_arguments_are_rejected_before_call():
    gateway = InProcessMCPGateway(_wrap_all_tools())
    response = gateway.call_tool(ToolCallRequest(tool_name="propagate_orbit", arguments={}))
    assert response.status == "invalid_arguments"
    assert "initial_state_dict" in (response.error or "")


def test_inprocess_gateway_preserves_structured_handler_failure():
    gateway = InProcessMCPGateway(
        {"fake": lambda: {"status": "error", "error": "boom"}},
        definitions=[
            {
                "name": "fake",
                "description": "test",
                "inputSchema": {"type": "object"},
            }
        ],
    )
    response = gateway.call_tool(ToolCallRequest(tool_name="fake"))
    assert response.status == "error"
    assert response.error == "boom"


def test_inprocess_gateway_invokes_explicit_injected_handler_without_schema():
    gateway = InProcessMCPGateway(
        {"failing_tool": lambda: (_ for _ in ()).throw(RuntimeError("boom"))}
    )
    response = gateway.call_tool(ToolCallRequest(tool_name="failing_tool"))
    assert response.status == "error"
    assert "boom" in (response.error or "")


def test_advertised_tool_definitions_have_wrapped_handlers():
    wrapped = _wrap_all_tools()
    report = get_tool_definition_report(get_tool_definitions(), wrapped)
    assert report["missing_handlers"] == []


def _new_stdio_gateway() -> StdioMCPGateway:
    return StdioMCPGateway(
        command=sys.executable,
        args=["-m", "aerospace_agent.mcp.server"],
        timeout=30,
    )


def test_stdio_gateway_initialize_list_call_and_close():
    gateway = _new_stdio_gateway()
    try:
        assert "check_engine_availability" in {tool.name for tool in gateway.list_tools()}
        result = gateway.call_tool(ToolCallRequest(tool_name="check_engine_availability"))
        assert result.status == "success"
    finally:
        gateway.close()
    assert gateway.closed is True
    gateway.close()


def test_sync_gateway_call_is_safe_while_caller_event_loop_is_running():
    async def caller():
        gateway = _new_stdio_gateway()
        try:
            return gateway.list_tools()
        finally:
            gateway.close()

    assert asyncio.run(caller())


def test_explicit_inprocess_fallback_emits_warning(mcp_settings):
    gateway, warnings = create_mcp_gateway(
        mcp_settings,
        allow_inprocess_fallback=True,
        force_stdio_failure=True,
    )
    assert isinstance(gateway, InProcessMCPGateway)
    assert warnings == ["MCP stdio unavailable; using explicit in-process fallback"]


def test_default_construction_raises_structured_unavailable_error(mcp_settings, monkeypatch):
    def unexpected_fallback():
        raise AssertionError("in-process fallback must remain opt-in")

    monkeypatch.setattr("aerospace_agent.mcp.server._wrap_all_tools", unexpected_fallback)
    with pytest.raises(MCPUnavailableError) as exc_info:
        create_mcp_gateway(mcp_settings, force_stdio_failure=True)
    assert exc_info.value.status == "tool_unavailable"
