from aerospace_agent.mcp.tool_schema import (
    get_tool_definition_report,
    validate_tool_definition,
)
from aerospace_agent.mcp.tools import TOOL_REGISTRY, get_tool_definitions


def test_validate_tool_definition_rejects_missing_core_mcp_fields():
    result = validate_tool_definition(
        {"name": "bad_tool", "inputSchema": {"type": "object"}}
    )

    assert result["valid"] is False
    assert "missing:description" in result["issues"]


def test_validate_space_tool_definition_requires_domain_metadata():
    result = validate_tool_definition(
        {
            "name": "space.bad",
            "description": "Bad space tool definition.",
            "inputSchema": {"type": "object"},
        }
    )

    assert result["valid"] is False
    assert "missing:outputSchema" in result["issues"]
    assert "missing:units" in result["issues"]
    assert "missing:validation_rules" in result["issues"]
    assert "missing:error_codes" in result["issues"]


def test_tool_definition_report_compares_definitions_to_registry_handlers():
    report = get_tool_definition_report(
        [
            {
                "name": "demo",
                "description": "Demo tool.",
                "inputSchema": {"type": "object"},
            }
        ],
        registry_keys={"demo", "handler_only"},
    )

    assert report["total"] == 1
    assert report["valid"] == 1
    assert report["invalid"] == 0
    assert report["missing_handlers"] == []
    assert report["handler_without_definition"] == ["handler_only"]


def test_registered_space_tool_definitions_pass_schema_audit():
    definitions = [
        item for item in get_tool_definitions()
        if item["name"].startswith("space.")
    ]
    report = get_tool_definition_report(definitions, registry_keys=TOOL_REGISTRY.keys())

    assert report["invalid_tools"] == []
    assert report["missing_handlers"] == []
