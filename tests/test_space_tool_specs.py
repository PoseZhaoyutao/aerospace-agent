from aerospace_agent.mcp.tools import (
    SPACE_TOOL_SPECS,
    TOOL_REGISTRY,
    get_space_tool_specs,
    get_tool_definitions,
)


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
}


def test_space_tool_specs_include_required_metadata_fields():
    specs = get_space_tool_specs()
    assert specs
    for spec in specs:
        assert REQUIRED_SPEC_FIELDS.issubset(spec)
        assert spec["name"].startswith("space.")
        assert spec["risk_level"] in {"P0", "P1", "P2"}
        assert isinstance(spec["validation_rules"], list)
        assert isinstance(spec["error_codes"], list)


def test_p0_space_tools_registered_and_exported_to_mcp_definitions():
    expected = {
        "space.check_environment",
        "space.convert_time",
        "space.propagate_orbit",
        "space.plot_orbit",
        "space.build_report",
        "space.reproduce_run",
    }

    assert expected.issubset(TOOL_REGISTRY)
    definition_names = {item["name"] for item in get_tool_definitions()}
    assert expected.issubset(definition_names)


def test_p1_space_tools_are_registered_as_explicit_unavailable_stubs():
    expected = {
        "space.load_spice_kernels",
        "space.transform_frame",
        "space.compute_attitude",
        "space.project_to_sensor",
        "space.query_star_catalog",
        "space.render_starfield",
        "space.compute_snr",
        "space.validate_truth",
    }

    spec_names = {item["name"] for item in SPACE_TOOL_SPECS}
    definition_names = {item["name"] for item in get_tool_definitions()}
    assert expected.issubset(spec_names)
    assert expected.issubset(definition_names)
    assert expected.issubset(TOOL_REGISTRY)

    for name in expected:
        result = TOOL_REGISTRY[name](example=True)
        assert result["status"] == "unavailable"
        assert result["error_code"] == "NOT_IMPLEMENTED"
        assert result["tool"] == name


def test_space_propagate_orbit_delegates_to_existing_propagator():
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
