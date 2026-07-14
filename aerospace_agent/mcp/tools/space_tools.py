"""Exact SpaceBasicTools contracts with truthful executable availability."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .environment_tools import check_engine_availability
from .frame_tools import transform_frame
from .propagation_tools import convert_orbit_representation, propagate_orbit
from .time_tools import convert_time
from .validation_tools import cross_validate_results


def _object(
    properties: dict[str, Any], required: list[str] | None = None
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


_STATE_SCHEMA = _object(
    {
        "epoch": {"type": "object"},
        "frame": {"type": "object"},
        "representation": {"type": "string"},
        "position_m": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "velocity_mps": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
        "elements": {"type": "object"},
    },
    ["epoch", "frame", "representation"],
)

_DEPENDENCIES = [
    {"name": "python", "version": sys.version.split()[0]},
    {"name": "zytAgent-internal", "version": "repository-snapshot"},
]


def _spec(
    name: str,
    description: str,
    input_schema: dict[str, Any],
    output_schema: dict[str, Any],
    *,
    status: str,
    units: str,
    frame: str,
    time_system: str,
    assumptions: list[str],
    risk_level: str = "P0",
    validator: str,
    validation_rules: list[str],
    error_codes: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "units": units,
        "frame": frame,
        "time_system": time_system,
        "assumptions": assumptions,
        "risk_level": risk_level,
        "required_files": [],
        "generated_artifacts": [],
        "validation_rules": validation_rules,
        "validator": validator,
        "dependencies": copy.deepcopy(_DEPENDENCIES),
        "adapter_sha256": "",  # bound to this file after construction
        "status": status,
        "error_codes": error_codes,
    }


SPACE_TOOL_SPECS: list[dict[str, Any]] = [
    _spec(
        "space.check_environment",
        "Report local astrodynamics engine availability without claiming missing engines.",
        _object({"engines": {"type": "array", "items": {"type": "string"}}, "timeout_seconds": {"type": "number"}}),
        _object({"status": {"type": "string"}, "engines": {"type": "object"}}, ["status", "engines"]),
        status="available", units="N/A", frame="N/A", time_system="N/A",
        assumptions=["Availability is a local runtime probe, not a numerical validation."],
        validator="space_contract.environment.v1",
        validation_rules=["Every requested engine has a structured availability record."],
        error_codes=["ENGINE_CHECK_FAILED"],
    ),
    _spec(
        "space.validate_state",
        "Validate canonical orbit state units, reference frame, epoch, and representation.",
        _object({"state": _STATE_SCHEMA}, ["state"]),
        _object({"status": {"type": "string"}, "normalized_state": {"type": "object"}, "violations": {"type": "array"}}, ["status", "violations"]),
        status="unavailable", units="SI: m, m/s, s", frame="Explicit canonical frame ID",
        time_system="Explicit epoch scale and format",
        assumptions=["No validation claim is permitted until a dedicated adapter and contract tests exist."],
        validator="unavailable",
        validation_rules=["Position, velocity, units, frame, and epoch are mandatory."],
        error_codes=["NOT_IMPLEMENTED", "INVALID_STATE"],
    ),
    _spec(
        "space.convert_time",
        "Convert an explicit time value between declared scales and formats.",
        _object({"value": {"type": ["string", "number"]}, "from_scale": {"type": "string"}, "from_format": {"type": "string"}, "to_scale": {"type": "string"}, "to_format": {"type": "string"}}, ["value", "from_scale", "from_format", "to_scale", "to_format"]),
        _object({"status": {"type": "string"}, "input": {"type": "object"}, "output": {}, "engine_used": {}, "notes": {"type": "string"}}, ["status", "input", "output"]),
        status="available", units="Seconds or declared calendar representation", frame="N/A",
        time_system="UTC/TAI/TT/TDB/ET",
        assumptions=["Accuracy and leap-second support depend on the reported engine_used value."],
        validator="space_contract.time_conversion.v1",
        validation_rules=["Source and target scale and format are explicit."],
        error_codes=["TIME_CONVERSION_FAILED", "UNSUPPORTED_TIME_SCALE"],
    ),
    _spec(
        "space.transform_frame",
        "Transform a canonical orbit state between explicitly declared frames.",
        _object({"state_dict": _STATE_SCHEMA, "target_frame": {"type": "string"}, "target_center": {"type": ["string", "null"]}}, ["state_dict", "target_frame"]),
        _object({"status": {"type": "string"}, "state": {"type": "object"}, "engine_used": {}, "frame_info": {"type": "object"}, "notes": {"type": "string"}}, ["status"]),
        status="available", units="SI: m, m/s, s", frame="Source and target frame explicit",
        time_system="State epoch is preserved",
        assumptions=["Analytic fallback has lower Earth-orientation fidelity than validated external libraries."],
        validator="space_contract.frame_transform.v1",
        validation_rules=["Source frame, target frame, center, and epoch remain auditable."],
        error_codes=["FRAME_TRANSFORM_FAILED", "MISSING_FRAME", "MISSING_EPOCH"],
    ),
    _spec(
        "space.convert_orbit_representation",
        "Convert canonical Cartesian and Keplerian orbit representations.",
        _object({"state_dict": _STATE_SCHEMA, "target_representation": {"type": "string"}, "mu": {"type": "number"}}, ["state_dict", "target_representation"]),
        _object({"status": {"type": "string"}, "state": {"type": "object"}, "source_representation": {"type": "string"}, "target_representation": {"type": "string"}, "mu": {"type": "number"}, "units": {}, "engine": {}}, ["status"]),
        status="available", units="SI: m, m/s, m^3/s^2", frame="Frame is preserved",
        time_system="Epoch is preserved",
        assumptions=["The supplied gravitational parameter applies to the declared central body."],
        validator="space_contract.orbit_representation.v1",
        validation_rules=["mu and output representation are reported."],
        error_codes=["ORBIT_CONVERSION_FAILED", "INVALID_REPRESENTATION"],
    ),
    _spec(
        "space.propagate_orbit",
        "Propagate a canonical Cartesian orbit state with an explicit force model.",
        _object({"initial_state_dict": _STATE_SCHEMA, "force_model_dict": {"type": "object"}, "duration_s": {"type": "number", "exclusiveMinimum": 0}, "output_step_s": {"type": ["number", "null"]}, "engine": {"type": "string"}}, ["initial_state_dict", "force_model_dict", "duration_s"]),
        _object({"status": {"type": "string"}, "state_history": {"type": "array"}, "metadata": {"type": "object"}}, ["status"]),
        status="available", units="SI: m, m/s, s", frame="Declared by initial_state_dict.frame",
        time_system="Declared by initial_state_dict.epoch",
        assumptions=["Builtin propagation supports the force-model subset reported in metadata."],
        validator="space_contract.propagation.v1",
        validation_rules=["Metadata reports engine, version, units, frame, and propagator type."],
        error_codes=["PROPAGATION_FAILED", "MISSING_FRAME", "MISSING_UNITS"],
    ),
    _spec(
        "space.cross_validate_results",
        "Compare independent propagation results against explicit thresholds.",
        _object({"task_spec": {"type": "object"}, "engines": {"type": ["array", "null"], "items": {"type": "string"}}, "existing_results": {"type": ["object", "null"]}, "thresholds": {"type": ["object", "null"]}}, ["task_spec"]),
        _object({"status": {"type": "string"}, "passed": {}, "confidence": {}, "position_error_m": {}, "velocity_error_mps": {}, "rms_position_error_m": {}, "rms_velocity_error_mps": {}, "event_time_error_s": {}, "difference_sources": {"type": "array"}, "thresholds": {"type": "object"}, "units": {"type": "string"}, "summary": {"type": "string"}, "per_engine_results": {"type": "object"}, "valid_engine_count": {"type": "integer"}}, ["status"]),
        status="available", units="m, m/s, s", frame="Inputs must share the compared frame",
        time_system="Histories must use aligned epochs",
        assumptions=["At least two independent valid results are required for a passing claim."],
        validator="space_contract.cross_validation.v1",
        validation_rules=["Thresholds and difference sources are reported."],
        error_codes=["INSUFFICIENT_RESULTS", "CROSS_VALIDATION_FAILED"],
    ),
]


def _success_envelope(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") in {"error", "unavailable", "failed"}:
        return result
    return {"status": "success", **result}


def _space_check_environment(**kwargs: Any) -> dict[str, Any]:
    return {"status": "success", "engines": check_engine_availability(**kwargs)}


def _space_convert_time(**kwargs: Any) -> dict[str, Any]:
    return _success_envelope(convert_time(**kwargs))


def _space_transform_frame(**kwargs: Any) -> dict[str, Any]:
    return _success_envelope(transform_frame(**kwargs))


def _space_convert_orbit_representation(**kwargs: Any) -> dict[str, Any]:
    return _success_envelope(convert_orbit_representation(**kwargs))


def _space_propagate_orbit(**kwargs: Any) -> dict[str, Any]:
    result = _success_envelope(propagate_orbit(**kwargs))
    if result.get("status") == "success":
        result.setdefault("metadata", {})["tool"] = "space.propagate_orbit"
    return result


def _space_cross_validate_results(**kwargs: Any) -> dict[str, Any]:
    return _success_envelope(cross_validate_results(**kwargs))


SPACE_TOOL_CALLS: dict[str, Callable[..., dict[str, Any]]] = {
    "space.check_environment": _space_check_environment,
    "space.convert_time": _space_convert_time,
    "space.transform_frame": _space_transform_frame,
    "space.convert_orbit_representation": _space_convert_orbit_representation,
    "space.propagate_orbit": _space_propagate_orbit,
    "space.cross_validate_results": _space_cross_validate_results,
}


_ADAPTER_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for _item in SPACE_TOOL_SPECS:
    _item["adapter_sha256"] = _ADAPTER_HASH
    if (_item["status"] == "available") != (_item["name"] in SPACE_TOOL_CALLS):
        raise RuntimeError("SpaceBasicTools availability/executor parity violation")


def get_space_tool_specs() -> list[dict[str, Any]]:
    return copy.deepcopy(SPACE_TOOL_SPECS)


def get_space_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": item["name"],
            "description": item["description"],
            "inputSchema": copy.deepcopy(item["input_schema"]),
            "outputSchema": copy.deepcopy(item["output_schema"]),
            "status": item["status"],
            "units": item["units"],
            "frame": item["frame"],
            "time_system": item["time_system"],
            "assumptions": list(item["assumptions"]),
            "risk_level": item["risk_level"],
            "validator": item["validator"],
            "dependencies": copy.deepcopy(item["dependencies"]),
            "adapter_sha256": item["adapter_sha256"],
            "required_files": list(item["required_files"]),
            "generated_artifacts": list(item["generated_artifacts"]),
            "validation_rules": list(item["validation_rules"]),
            "error_codes": list(item["error_codes"]),
        }
        for item in SPACE_TOOL_SPECS
    ]


def register_space_tools(registry: MutableMapping[str, Callable[..., dict[str, Any]]]) -> None:
    registry.update(SPACE_TOOL_CALLS)


__all__ = [
    "SPACE_TOOL_CALLS",
    "SPACE_TOOL_SPECS",
    "get_space_tool_definitions",
    "get_space_tool_specs",
    "register_space_tools",
]
