import hashlib
from pathlib import Path

from jsonschema import validate

from aerospace_agent.mcp.tools import SPACE_TOOL_SPECS, TOOL_REGISTRY, get_space_tool_specs


REQUIRED_SPEC_FIELDS = {
    "name",
    "description",
    "input_schema",
    "output_schema",
    "units",
    "frame",
    "time_system",
    "risk_level",
    "required_files",
    "generated_artifacts",
    "validation_rules",
    "error_codes",
    "status",
    "assumptions",
    "validator",
    "dependencies",
    "adapter_sha256",
}

EXACT_SPACE_BASIC_TOOLS = {
    "space.check_environment",
    "space.validate_state",
    "space.convert_time",
    "space.transform_frame",
    "space.convert_orbit_representation",
    "space.propagate_orbit",
    "space.cross_validate_results",
}


def test_space_tool_specs_include_required_metadata_fields():
    specs = get_space_tool_specs()
    assert {item["name"] for item in specs} == EXACT_SPACE_BASIC_TOOLS
    assert len(specs) == 7
    for spec in specs:
        assert REQUIRED_SPEC_FIELDS.issubset(spec)
        assert spec["name"].startswith("space.")
        assert spec["risk_level"] in {"P0", "P1", "P2"}
        assert isinstance(spec["validation_rules"], list)
        assert isinstance(spec["error_codes"], list)
        assert spec["status"] in {"available", "unavailable"}
        assert spec["input_schema"]["type"] == "object"
        assert spec["input_schema"].get("additionalProperties") is False
        assert spec["output_schema"]["type"] == "object"
        assert spec["units"]
        assert spec["frame"]
        assert spec["time_system"]
        assert spec["assumptions"]
        assert spec["validator"]
        assert all("name" in item and "version" in item for item in spec["dependencies"])
        assert len(spec["adapter_sha256"]) == 64
        int(spec["adapter_sha256"], 16)


def test_only_available_space_tools_have_executors_and_hash_binds_adapter():
    available = {item["name"] for item in SPACE_TOOL_SPECS if item["status"] == "available"}
    registered = EXACT_SPACE_BASIC_TOOLS.intersection(TOOL_REGISTRY)
    assert available == registered
    unavailable = EXACT_SPACE_BASIC_TOOLS - available
    assert unavailable.isdisjoint(TOOL_REGISTRY)

    from aerospace_agent.mcp.tools import space_tools

    adapter_hash = hashlib.sha256(Path(space_tools.__file__).read_bytes()).hexdigest()
    assert {item["adapter_sha256"] for item in SPACE_TOOL_SPECS} == {adapter_hash}


def test_space_propagate_orbit_delegates_to_existing_propagator():
    if "space.propagate_orbit" not in TOOL_REGISTRY:
        spec = next(item for item in SPACE_TOOL_SPECS if item["name"] == "space.propagate_orbit")
        assert spec["status"] == "unavailable"
        return
    result = TOOL_REGISTRY["space.propagate_orbit"](
        initial_state_dict={
            "epoch": {
                "value": "2026-01-01T00:00:00",
                "scale": "UTC",
                "format": "ISO",
            },
            "frame": {
                "name": "GCRF",
                "center": "Earth",
                "realization": "IERS2010",
            },
            "representation": "cartesian",
            "position_m": [6778137.0, 0.0, 0.0],
            "velocity_mps": [0.0, 7668.558175407055, 0.0],
        },
        force_model_dict={"gravity": "point_mass"},
        duration_s=1200.0,
        output_step_s=600.0,
        engine="builtin",
    )

    assert "state_history" in result
    assert len(result["state_history"]) == 3
    assert result["metadata"]["engine"] == "builtin"
    assert result["metadata"]["frame"] == "GCRF"


def test_every_available_space_adapter_passes_its_declared_output_contract():
    state = {
        "epoch": {"value": "2026-01-01T00:00:00", "scale": "UTC", "format": "ISO"},
        "frame": {"name": "GCRF", "center": "Earth", "realization": "IERS2010"},
        "representation": "cartesian",
        "position_m": [6778137.0, 0.0, 0.0],
        "velocity_mps": [0.0, 7668.558175407055, 0.0],
    }
    existing = {
        "reference": {"status": "success", "final_state": state},
        "candidate": {"status": "success", "final_state": state},
    }
    calls = {
        "space.check_environment": {"engines": ["not-a-real-engine"]},
        "space.convert_time": {
            "value": "2026-01-01T00:00:00",
            "from_scale": "UTC",
            "from_format": "ISO",
            "to_scale": "UTC",
            "to_format": "ISO",
        },
        "space.transform_frame": {"state_dict": state, "target_frame": "GCRF"},
        "space.convert_orbit_representation": {
            "state_dict": state,
            "target_representation": "cartesian",
        },
        "space.propagate_orbit": {
            "initial_state_dict": state,
            "force_model_dict": {"gravity": "point_mass"},
            "duration_s": 1.0,
            "engine": "builtin",
        },
        "space.cross_validate_results": {
            "task_spec": {},
            "existing_results": existing,
        },
    }

    for spec in SPACE_TOOL_SPECS:
        if spec["status"] != "available":
            continue
        result = TOOL_REGISTRY[spec["name"]](**calls[spec["name"]])
        validate(result, spec["output_schema"])
        assert result["status"] in {"success", "error", "unavailable"}
