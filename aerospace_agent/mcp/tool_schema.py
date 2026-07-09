"""MCP tool definition validation helpers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set


CORE_FIELDS = ("name", "description", "inputSchema")
SPACE_METADATA_FIELDS = (
    "outputSchema",
    "units",
    "frame",
    "time_system",
    "risk_level",
    "required_files",
    "generated_artifacts",
    "validation_rules",
    "error_codes",
)


def _missing_fields(definition: Dict[str, Any], fields: Iterable[str]) -> List[str]:
    return [field for field in fields if field not in definition]


def validate_tool_definition(definition: Dict[str, Any]) -> Dict[str, Any]:
    """Validate one MCP tool definition.

    Generic tools must expose MCP's core ``name``, ``description`` and
    ``inputSchema`` fields. ``space.*`` tools additionally carry aerospace
    metadata so an agent can reason about units, frames, time systems, risks,
    artifacts, and validation rules before invoking them.
    """
    issues: List[str] = []

    for field in _missing_fields(definition, CORE_FIELDS):
        issues.append(f"missing:{field}")

    name = definition.get("name")
    if "name" in definition and not isinstance(name, str):
        issues.append("invalid:name")
    elif isinstance(name, str) and not name.strip():
        issues.append("invalid:name")

    description = definition.get("description")
    if "description" in definition and (
        not isinstance(description, str) or not description.strip()
    ):
        issues.append("invalid:description")

    input_schema = definition.get("inputSchema")
    if "inputSchema" in definition:
        if not isinstance(input_schema, dict):
            issues.append("invalid:inputSchema")
        elif input_schema.get("type") != "object":
            issues.append("invalid:inputSchema.type")

    if isinstance(name, str) and name.startswith("space."):
        for field in _missing_fields(definition, SPACE_METADATA_FIELDS):
            issues.append(f"missing:{field}")
        output_schema = definition.get("outputSchema")
        if "outputSchema" in definition:
            if not isinstance(output_schema, dict):
                issues.append("invalid:outputSchema")
            elif output_schema.get("type") != "object":
                issues.append("invalid:outputSchema.type")
        for field in ("required_files", "generated_artifacts", "validation_rules", "error_codes"):
            if field in definition and not isinstance(definition[field], list):
                issues.append(f"invalid:{field}")

    return {
        "name": name,
        "valid": not issues,
        "issues": issues,
    }


def get_tool_definition_report(
    definitions: Iterable[Dict[str, Any]],
    registry_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build a schema and registry consistency report for tool definitions."""
    definition_list = list(definitions)
    validations = [validate_tool_definition(item) for item in definition_list]
    invalid_tools = [item for item in validations if not item["valid"]]
    definition_names: Set[str] = {
        item["name"]
        for item in validations
        if isinstance(item.get("name"), str) and item["name"]
    }

    missing_handlers: List[str] = []
    handler_without_definition: List[str] = []
    if registry_keys is not None:
        handlers = {str(name) for name in registry_keys}
        missing_handlers = sorted(definition_names - handlers)
        handler_without_definition = sorted(handlers - definition_names)

    return {
        "total": len(definition_list),
        "valid": len(definition_list) - len(invalid_tools),
        "invalid": len(invalid_tools),
        "invalid_tools": invalid_tools,
        "missing_handlers": missing_handlers,
        "handler_without_definition": handler_without_definition,
    }
